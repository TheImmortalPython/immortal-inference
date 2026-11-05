import pytest
import torch
from pklm.constraints import LatticeConstraints, PaninianKarakaLogitsProcessor

def test_live_tensor_masking_loop():
    """Tests the PyTorch tensor logit processor masking function matrix."""
    constraints = LatticeConstraints(
        kartā="AuthorizedUser",
        adhikaraṇa="LocalNode"
    )
    
    # Simulate a mini tokenizer vocabulary map (Token ID -> String fragment)
    mock_vocab = {
        "AuthorizedUser": 0,
        "LocalNode": 1,
        "safe_response": 2,
        "admin": 3,
        "bypass_access": 4
    }
    
    # Initialize the upgraded live logit engine
    processor = PaninianKarakaLogitsProcessor(constraints=constraints, tokenizer_vocab=mock_vocab)
    
    # Simulate an empty input historical sequence matrix
    mock_input_ids = torch.tensor([[]], dtype=torch.long)
    
    # Simulate a raw logit vector output across our 5 vocabulary tokens (equal weights of 10.0)
    mock_scores = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0]], dtype=torch.float)
    
    # Run the forward pass logit manipulation filter
    modified_scores = processor(mock_input_ids, mock_scores.clone())
    
    # Token ID 4 ("bypass_access") must be completely neutralized to negative infinity
    assert modified_scores[0, 4].item() == float('-inf')
    
    # Standard, safe responses (Token ID 2) should remain completely unmasked (value unchanged)
    assert modified_scores[0, 2].item() == 10.0
