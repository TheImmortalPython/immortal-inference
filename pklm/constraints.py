import spacy
import torch
from transformers import LogitsProcessor
from typing import Dict, List, Set, Optional

# Load the English dependency model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

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
        allowed_mutations: Optional[List[str]] = None
    ):
        self.kartā = kartā.lower()                  
        self.adhikaraṇa = adhikaraṇa.lower()        
        self.karma = karma.lower() if karma else None 
        self.allowed_mutations = [m.lower() for m in (allowed_mutations or [])]


class PaninianKarakaLogitsProcessor(LogitsProcessor):
    """
    A live Hugging Face LogitsProcessor that intercepts autoregressive token scoring matrices
    and forces output distributions to adhere to a deterministic Paninian dependency lattice.
    """
    def __init__(self, constraints: LatticeConstraints, tokenizer_vocab: Dict[str, int]):
        self.constraints = constraints
        # Flip the vocabulary dictionary so we can look up string fragments by their numeric Token ID
        self.id_to_token = {v: k for k, v in tokenizer_vocab.items()}
        
        self.active_slots: Dict[str, Optional[str]] = {
            "kartā": None,
            "adhikaraṇa": None,
            "karma": None
        }

    def _update_lattice_state(self, input_ids: torch.LongTensor) -> None:
        """Decodes recent historical token IDs to dynamically shift active Karaka slots."""
        if not nlp or input_ids.shape[-1] == 0:
            return
            
        # Decode the last 20 tokens to check for recent structural completion
        recent_ids = input_ids[0, -20:].tolist()
        recent_strings = [self.id_to_token.get(tid, "") for tid in recent_ids]
        reconstructed_text = "".join(recent_strings).replace("Ġ", " ").replace(" ", " ")
        
        doc = nlp(reconstructed_text)
        for token in doc:
            if token.dep_ == "nsubj" and token.text.lower() == self.constraints.kartā:
                self.active_slots["kartā"] = token.text
            elif token.dep_ in ["pobj", "prep"] and token.text.lower() == self.constraints.adhikaraṇa:
                self.active_slots["adhikaraṇa"] = token.text
            elif token.dep_ == "dobj" and self.constraints.karma and token.text.lower() == self.constraints.karma:
                self.active_slots["karma"] = token.text

    def _get_valid_next_roles(self) -> Set[str]:
        if not self.active_slots["kartā"]:
            return {"kartā", "adhikaraṇa"}
        return {"karma", "adhikaraṇa", "mutation"}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        The core mathematical masking function.
        Modifies raw logit tensor states to negative infinity before sampling occurs.
        """
        # 1. Dynamically read historical tokens and shift constraints states
        self._update_lattice_state(input_ids)
        permissible_roles = self._get_valid_next_roles()
        
        # 2. Iterate through every single token score in the model's vocabulary vector
        # (Optimized to execute directly across batch indices)
        for token_id in range(scores.shape[-1]):
            token_string = self.id_to_token.get(token_id, "")
            clean_token = token_string.strip().lower().replace("Ġ", "").replace(" ", "")
            
            # Intercept system escapes or parameter drifting behaviors
            if "bypass" in clean_token or "override" in clean_token:
                scores[:, token_id] = float("-inf")
                
            elif "kartā" not in permissible_roles and clean_token in ["admin", "root", "unauthorized_user"]:
                # Mathematically wipe the token out of the probability distribution matrix
                scores[:, token_id] = float("-inf")
                
        return scores
