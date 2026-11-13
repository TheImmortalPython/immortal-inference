import re
import spacy
import torch
from transformers import LogitsProcessor
from typing import Dict, List, Set, Optional

# Load the English dependency model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;:\n]")


class LatticeConstraints:
    """
    Defines the structural, relational constraints based on Pāṇinian Kāraka rules.
    Maps system boundaries across all six Kāraka roles to structural natural language targets.
    """
    def __init__(
        self,
        kartā: str,
        adhikaraṇa: str,
        karma: Optional[str] = None,
        karaṇa: Optional[str] = None,
        sampradāna: Optional[str] = None,
        apādāna: Optional[str] = None,
        allowed_mutations: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        restricted_karta_tokens: Optional[List[str]] = None,
        restricted_karma_tokens: Optional[List[str]] = None,
        restricted_karana_tokens: Optional[List[str]] = None,
    ):
        # Core 6 Pāṇinian Kārakas
        self.kartā = kartā.lower()
        self.adhikaraṇa = adhikaraṇa.lower()
        self.karma = karma.lower() if karma else None
        self.karaṇa = karaṇa.lower() if karaṇa else None
        self.sampradāna = sampradāna.lower() if sampradāna else None
        self.apādāna = apādāna.lower() if apādāna else None

        self.allowed_mutations = [m.lower() for m in (allowed_mutations or [])]
        self.forbidden_keywords = [
            w.lower() for w in (forbidden_keywords or ["bypass", "override"])
        ]
        self.restricted_karta_tokens = [
            w.lower() for w in (restricted_karta_tokens or ["admin", "root", "unauthorized_user"])
        ]
        self.restricted_karma_tokens = [
            w.lower() for w in (restricted_karma_tokens or ["production_db", "system_kernel", "vault"])
        ]
        self.restricted_karana_tokens = [
            w.lower() for w in (restricted_karana_tokens or ["master_key", "sudo_token", "raw_socket"])
        ]


class PaninianKarakaLogitsProcessor(LogitsProcessor):
    """
    A live Hugging Face LogitsProcessor that intercepts autoregressive token scoring matrices
    and forces output distributions to adhere to a deterministic multi-role Pāṇinian dependency lattice.

    Features:
      1. Full 6-Kāraka relational tracking (Kartā, Karma, Karaṇa, Sampradāna, Apādāna, Adhikaraṇa).
      2. Conditional role-gating: privileged Karma/Karaṇa tokens remain masked until Kartā + Adhikaraṇa are verified.
      3. Pre-indexed boolean PyTorch tensor masks for vectorized GPU masked_fill_ execution.
      4. Rolling sub-word tail buffer to block split-token bypasses across all restricted role sets.
    """
    def __init__(self, constraints: LatticeConstraints, tokenizer_vocab: Dict[str, int]):
        self.constraints = constraints
        self.id_to_token: Dict[int, str] = {v: k for k, v in tokenizer_vocab.items()}
        self.vocab_size: int = max(tokenizer_vocab.values(), default=-1) + 1

        self.active_slots: Dict[str, Optional[str]] = {
            "kartā": None,
            "karma": None,
            "karaṇa": None,
            "sampradāna": None,
            "apādāna": None,
            "adhikaraṇa": None,
        }
        self._last_parsed_clause: str = ""
        self._last_clause_index: int = 0

        # Pre-compute normalized token strings and prefix lookup tables
        self._clean_tokens: List[str] = [""] * self.vocab_size
        self._starts_new_word: List[bool] = [False] * self.vocab_size

        all_protected_words = (
            self.constraints.forbidden_keywords
            + self.constraints.restricted_karta_tokens
            + self.constraints.restricted_karma_tokens
            + self.constraints.restricted_karana_tokens
        )
        self._max_protected_len: int = max((len(w) for w in all_protected_words), default=0)

        # Build prefix sets for sub-word split detection
        self._global_prefixes: Set[str] = self._build_prefixes(self.constraints.forbidden_keywords)
        self._karta_prefixes: Set[str] = self._build_prefixes(self.constraints.restricted_karta_tokens)
        self._karma_prefixes: Set[str] = self._build_prefixes(self.constraints.restricted_karma_tokens)
        self._karana_prefixes: Set[str] = self._build_prefixes(self.constraints.restricted_karana_tokens)

        # Pre-index static boolean masks on CPU (moved lazily to target device on first forward pass)
        global_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        karta_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        karma_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        karana_mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        for tok_str, tid in tokenizer_vocab.items():
            if tid < 0 or tid >= self.vocab_size:
                continue
            starts_space = tok_str.startswith((" ", "Ġ", "▁"))
            clean = self._normalize_fragment(tok_str)
            self._clean_tokens[tid] = clean
            self._starts_new_word[tid] = starts_space

            if clean:
                if any(kw in clean for kw in self.constraints.forbidden_keywords):
                    global_mask[tid] = True
                if clean in self.constraints.restricted_karta_tokens:
                    karta_mask[tid] = True
                if clean in self.constraints.restricted_karma_tokens:
                    karma_mask[tid] = True
                if clean in self.constraints.restricted_karana_tokens:
                    karana_mask[tid] = True

        self._global_block_mask: torch.BoolTensor = global_mask
        self._karta_block_mask: torch.BoolTensor = karta_mask
        self._karma_block_mask: torch.BoolTensor = karma_mask
        self._karana_block_mask: torch.BoolTensor = karana_mask
        self._subword_mask_cache: Dict[tuple, torch.BoolTensor] = {}

    @staticmethod
    def _normalize_fragment(token_str: str) -> str:
        return token_str.replace("Ġ", " ").replace("▁", " ").strip().lower()

    @staticmethod
    def _build_prefixes(words: List[str]) -> Set[str]:
        prefixes: Set[str] = set()
        for word in words:
            for i in range(1, len(word)):
                prefixes.add(word[:i])
        return prefixes

    def _ensure_device(self, device: torch.device) -> None:
        if self._global_block_mask.device != device:
            self._global_block_mask = self._global_block_mask.to(device)
            self._karta_block_mask = self._karta_block_mask.to(device)
            self._karma_block_mask = self._karma_block_mask.to(device)
            self._karana_block_mask = self._karana_block_mask.to(device)
            self._subword_mask_cache.clear()

    def _reconstruct_recent_text(self, input_ids: torch.LongTensor, window: int = 48) -> str:
        recent_ids = input_ids[0, -window:].tolist()
        raw = "".join(self.id_to_token.get(tid, "") for tid in recent_ids)
        return raw.replace("Ġ", " ").replace("▁", " ")

    def _extract_active_clause(self, text: str) -> str:
        matches = list(CLAUSE_BOUNDARY_RE.finditer(text))
        if not matches:
            return text.strip()
        last_boundary = matches[-1].end()
        return text[last_boundary:].strip()

    def _update_lattice_state(self, recent_text: str) -> None:
        """Parses only the active clause boundary to update all 6 Kāraka slots incrementally."""
        if not nlp or not recent_text:
            return

        # Reset active slots when a new sentence/clause begins
        boundary_count = len(CLAUSE_BOUNDARY_RE.findall(recent_text))
        if boundary_count != self._last_clause_index:
            self._last_clause_index = boundary_count
            self.active_slots = {
                "kartā": None,
                "karma": None,
                "karaṇa": None,
                "sampradāna": None,
                "apādāna": None,
                "adhikaraṇa": None,
            }

        active_clause = self._extract_active_clause(recent_text)
        if not active_clause or active_clause == self._last_parsed_clause:
            return

        self._last_parsed_clause = active_clause
        doc = nlp(active_clause)
        for token in doc:
            tok_lower = token.text.lower()
            prep_head = token.head.text.lower() if token.dep_ == "pobj" else ""

            if token.dep_ in ("nsubj", "nsubjpass") and tok_lower == self.constraints.kartā:
                self.active_slots["kartā"] = token.text
            elif token.dep_ in ("dobj", "obj") and self.constraints.karma and tok_lower == self.constraints.karma:
                self.active_slots["karma"] = token.text
            elif token.dep_ == "pobj" and prep_head in ("with", "via", "using") and self.constraints.karaṇa and tok_lower == self.constraints.karaṇa:
                self.active_slots["karaṇa"] = token.text
            elif token.dep_ in ("dative", "pobj") and prep_head in ("to", "for") and self.constraints.sampradāna and tok_lower == self.constraints.sampradāna:
                self.active_slots["sampradāna"] = token.text
            elif token.dep_ == "pobj" and prep_head == "from" and self.constraints.apādāna and tok_lower == self.constraints.apādāna:
                self.active_slots["apādāna"] = token.text
            elif token.dep_ in ("pobj", "prep") and tok_lower == self.constraints.adhikaraṇa:
                self.active_slots["adhikaraṇa"] = token.text

    def _get_valid_next_roles(self) -> Set[str]:
        """
        Determines which Kāraka roles are syntactically and securely permissible
        based on already-bound slots in the active clause.
        """
        # Phase 1: Require authorized Kartā (Agent) before unlocking downstream roles
        if not self.active_slots["kartā"]:
            return {"kartā", "adhikaraṇa", "apādāna"}

        # Phase 2: Kartā is verified, but Adhikaraṇa (Locus) is not yet bound
        if not self.active_slots["adhikaraṇa"]:
            return {"adhikaraṇa", "apādāna", "sampradāna"}

        # Phase 3: Both Kartā + Adhikaraṇa verified -> unlock Karma, Karaṇa, and mutations
        return {"karma", "karaṇa", "sampradāna", "apādāna", "adhikaraṇa", "mutation"}

    def _get_trailing_word_fragment(self, recent_text: str) -> str:
        if not recent_text or recent_text[-1].isspace():
            return ""
        parts = re.split(r"[\s.!?;:\n\"'()\[\]{}]+", recent_text)
        tail = parts[-1].lower() if parts else ""
        return tail[-self._max_protected_len :] if self._max_protected_len > 0 else ""

    def _get_dynamic_subword_mask(
        self,
        tail: str,
        block_karta: bool,
        block_karma: bool,
        block_karana: bool,
        device: torch.device,
    ) -> Optional[torch.BoolTensor]:
        if not tail:
            return None

        in_global = tail in self._global_prefixes
        in_karta = block_karta and (tail in self._karta_prefixes)
        in_karma = block_karma and (tail in self._karma_prefixes)
        in_karana = block_karana and (tail in self._karana_prefixes)

        if not (in_global or in_karta or in_karma or in_karana):
            return None

        cache_key = (tail, block_karta, block_karma, block_karana)
        cached = self._subword_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        dyn_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for tid in range(self.vocab_size):
            if self._starts_new_word[tid]:
                continue
            frag = self._clean_tokens[tid]
            if not frag:
                continue
            combined = tail + frag
            if any(kw in combined for kw in self.constraints.forbidden_keywords):
                dyn_mask[tid] = True
            elif block_karta and combined in self.constraints.restricted_karta_tokens:
                dyn_mask[tid] = True
            elif block_karma and combined in self.constraints.restricted_karma_tokens:
                dyn_mask[tid] = True
            elif block_karana and combined in self.constraints.restricted_karana_tokens:
                dyn_mask[tid] = True

        dyn_mask = dyn_mask.to(device)
        self._subword_mask_cache[cache_key] = dyn_mask
        return dyn_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Vectorized GPU logit masking across all 6 Pāṇinian Kārakas with sub-word
        boundary protection and incremental clause-level dependency parsing.
        """
        vocab_dim = scores.shape[-1]
        self._ensure_device(scores.device)

        # 1. Reconstruct recent window and update all 6 Kāraka slots on active clause only
        recent_text = self._reconstruct_recent_text(input_ids) if input_ids.shape[-1] > 0 else ""
        self._update_lattice_state(recent_text)
        permissible_roles = self._get_valid_next_roles()

        block_karta = "kartā" not in permissible_roles
        block_karma = "karma" not in permissible_roles
        block_karana = "karaṇa" not in permissible_roles

        # 2. Apply pre-indexed static boolean tensor masks in vectorized GPU ops
        active_mask = self._global_block_mask[:vocab_dim]
        if block_karta:
            active_mask = active_mask | self._karta_block_mask[:vocab_dim]
        if block_karma:
            active_mask = active_mask | self._karma_block_mask[:vocab_dim]
        if block_karana:
            active_mask = active_mask | self._karana_block_mask[:vocab_dim]

        # 3. Check rolling sub-word tail buffer when trailing chars match any active protected prefix
        tail = self._get_trailing_word_fragment(recent_text)
        dyn_mask = self._get_dynamic_subword_mask(
            tail, block_karta, block_karma, block_karana, scores.device
        )
        if dyn_mask is not None:
            active_mask = active_mask | dyn_mask[:vocab_dim]

        scores.masked_fill_(active_mask.unsqueeze(0), float("-inf"))
        return scores
