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
    Defines the structural, relational constraints based on Paninian Karaka rules.
    Maps system boundaries to structural natural language targets.
    """
    def __init__(
        self,
        kartā: str,
        adhikaraṇa: str,
        karma: Optional[str] = None,
        allowed_mutations: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        restricted_karta_tokens: Optional[List[str]] = None,
    ):
        self.kartā = kartā.lower()
        self.adhikaraṇa = adhikaraṇa.lower()
        self.karma = karma.lower() if karma else None
        self.allowed_mutations = [m.lower() for m in (allowed_mutations or [])]
        self.forbidden_keywords = [
            w.lower() for w in (forbidden_keywords or ["bypass", "override"])
        ]
        self.restricted_karta_tokens = [
            w.lower() for w in (restricted_karta_tokens or ["admin", "root", "unauthorized_user"])
        ]


class PaninianKarakaLogitsProcessor(LogitsProcessor):
    """
    A live Hugging Face LogitsProcessor that intercepts autoregressive token scoring matrices
    and forces output distributions to adhere to a deterministic Paninian dependency lattice.

    Features:
      1. Pre-indexed boolean PyTorch tensor masks for vectorized GPU masked_fill_ execution.
      2. Rolling sub-word tail buffer to block split-token bypasses (e.g., "adm" + "in").
      3. Incremental clause-level spaCy parsing with deterministic lexical fallback.
    """
    def __init__(self, constraints: LatticeConstraints, tokenizer_vocab: Dict[str, int]):
        self.constraints = constraints
        self.id_to_token: Dict[int, str] = {v: k for k, v in tokenizer_vocab.items()}
        self.vocab_size: int = max(tokenizer_vocab.values(), default=-1) + 1

        self.active_slots: Dict[str, Optional[str]] = {
            "kartā": None,
            "adhikaraṇa": None,
            "karma": None,
        }
        self._last_parsed_clause: str = ""
        self._last_clause_index: int = 0

        # Pre-compute normalized token strings and prefix lookup tables
        self._clean_tokens: List[str] = [""] * self.vocab_size
        self._starts_new_word: List[bool] = [False] * self.vocab_size
        self._max_protected_len: int = max(
            (len(w) for w in (self.constraints.forbidden_keywords + self.constraints.restricted_karta_tokens)),
            default=0,
        )

        # Build prefix sets for sub-word split detection
        self._global_prefixes: Set[str] = self._build_prefixes(self.constraints.forbidden_keywords)
        self._karta_prefixes: Set[str] = self._build_prefixes(self.constraints.restricted_karta_tokens)

        # Pre-index static boolean masks on CPU (moved lazily to target device on first forward pass)
        global_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        karta_mask = torch.zeros(self.vocab_size, dtype=torch.bool)

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

        self._global_block_mask: torch.BoolTensor = global_mask
        self._karta_block_mask: torch.BoolTensor = karta_mask
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
            self._subword_mask_cache.clear()

    def _reconstruct_recent_text(self, input_ids: torch.LongTensor, window: int = 32) -> str:
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
        """
        Parses only the active clause boundary to update Karaka slots incrementally.
        Uses spaCy dependency parsing when available, with a deterministic lexical
        fallback if the spaCy model is not installed in the runtime environment.
        """
        if not recent_text:
            return

        # Always track clause boundaries and reset active slots across sentences
        boundary_count = len(CLAUSE_BOUNDARY_RE.findall(recent_text))
        if boundary_count != self._last_clause_index:
            self._last_clause_index = boundary_count
            self.active_slots = {"kartā": None, "adhikaraṇa": None, "karma": None}
            self._last_parsed_clause = ""

        active_clause = self._extract_active_clause(recent_text)
        if not active_clause or active_clause == self._last_parsed_clause:
            return

        self._last_parsed_clause = active_clause

        if nlp is not None:
            doc = nlp(active_clause)
            for token in doc:
                tok_lower = token.text.lower()
                if token.dep_ == "nsubj" and tok_lower == self.constraints.kartā:
                    self.active_slots["kartā"] = token.text
                elif token.dep_ in ("pobj", "prep") and tok_lower == self.constraints.adhikaraṇa:
                    self.active_slots["adhikaraṇa"] = token.text
                elif token.dep_ == "dobj" and self.constraints.karma and tok_lower == self.constraints.karma:
                    self.active_slots["karma"] = token.text
        else:
            # Deterministic lexical fallback when en_core_web_sm is unavailable
            words = re.findall(r"[a-zA-Z0-9_]+", active_clause)
            for word in words:
                w_lower = word.lower()
                if w_lower == self.constraints.kartā:
                    self.active_slots["kartā"] = word
                elif w_lower == self.constraints.adhikaraṇa:
                    self.active_slots["adhikaraṇa"] = word
                elif self.constraints.karma and w_lower == self.constraints.karma:
                    self.active_slots["karma"] = word

    def _get_valid_next_roles(self) -> Set[str]:
        if not self.active_slots["kartā"]:
            return {"kartā", "adhikaraṇa"}
        return {"karma", "adhikaraṇa", "mutation"}

    def _get_trailing_word_fragment(self, recent_text: str) -> str:
        if not recent_text or recent_text[-1].isspace():
            return ""
        parts = re.split(r"[\s.!?;:\n\"'()\[\]{}]+", recent_text)
        tail = parts[-1].lower() if parts else ""
        if self._max_protected_len <= 0:
            return ""
        start_idx = max(0, len(tail) - self._max_protected_len)
        return tail[start_idx:]

    def _get_dynamic_subword_mask(
        self, tail: str, block_karta_roles: bool, device: torch.device
    ) -> Optional[torch.BoolTensor]:
        if not tail:
            return None

        in_global = tail in self._global_prefixes
        in_karta = block_karta_roles and (tail in self._karta_prefixes)
        if not in_global and not in_karta:
            return None

        cache_key = (tail, block_karta_roles)
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
            elif block_karta_roles and combined in self.constraints.restricted_karta_tokens:
                dyn_mask[tid] = True

        dyn_mask = dyn_mask.to(device)
        self._subword_mask_cache[cache_key] = dyn_mask
        return dyn_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Vectorized GPU logit masking with sub-word boundary protection and
        incremental clause-level dependency parsing.
        """
        vocab_dim = scores.shape[-1]
        self._ensure_device(scores.device)

        # 1. Reconstruct recent window and update Karaka slots on active clause only
        recent_text = self._reconstruct_recent_text(input_ids) if input_ids.shape[-1] > 0 else ""
        self._update_lattice_state(recent_text)
        permissible_roles = self._get_valid_next_roles()
        block_karta_roles = "kartā" not in permissible_roles
       
        # 2. Apply pre-indexed static boolean tensor masks in vectorized GPU ops
        active_mask = self._global_block_mask.narrow(0, 0, vocab_dim)
        if block_karta_roles:
        active_mask = active_mask | self._karta_block_mask.narrow(0, 0, vocab_dim)
            
        # 3. Check rolling sub-word tail buffer only when trailing chars match a protected prefix
        tail = self._get_trailing_word_fragment(recent_text)
        dyn_mask = self._get_dynamic_subword_mask(tail, block_karta_roles, scores.device)
        if dyn_mask is not None:
        active_mask = active_mask | dyn_mask.narrow(0, 0, vocab_dim)
        scores.masked_fill_(active_mask.unsqueeze(0), float("-inf"))
        return scores
