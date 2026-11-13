import pytest
import torch
from pklm.constraints import LatticeConstraints, PaninianKarakaLogitsProcessor


@pytest.fixture
def base_constraints():
    return LatticeConstraints(
        kartā="AuthorizedUser",
        adhikaraṇa="LocalNode",
        karma="task",
    )


@pytest.fixture
def extended_vocab():
    return {
        "AuthorizedUser": 0,
        "LocalNode": 1,
        "safe_response": 2,
        "admin": 3,
        "bypass_access": 4,
        "adm": 5,
        "in": 6,
        "by": 7,
        "pass": 8,
        " ": 9,
        ".": 10,
        "executes": 11,
        "task": 12,
    }


def test_live_tensor_masking_loop(base_constraints):
    """Tests the PyTorch tensor logit processor masking function matrix."""
    mock_vocab = {
        "AuthorizedUser": 0,
        "LocalNode": 1,
        "safe_response": 2,
        "admin": 3,
        "bypass_access": 4,
    }

    processor = PaninianKarakaLogitsProcessor(
        constraints=base_constraints,
        tokenizer_vocab=mock_vocab,
    )

    mock_input_ids = torch.zeros((1, 0), dtype=torch.long)
    mock_scores = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0]], dtype=torch.float)

    modified_scores = processor(mock_input_ids, mock_scores.clone())

    # Token ID 4 ("bypass_access") must be completely neutralized to negative infinity
    assert modified_scores[0, 4].item() == float("-inf")

    # Standard, safe responses (Token ID 2) should remain completely unmasked
    assert modified_scores[0, 2].item() == 10.0


def test_preindexed_masks_constructed_at_init(base_constraints, extended_vocab):
    """Verifies static boolean masks are pre-computed at initialization."""
    processor = PaninianKarakaLogitsProcessor(
        constraints=base_constraints,
        tokenizer_vocab=extended_vocab,
    )

    assert processor._global_block_mask.dtype == torch.bool
    assert processor._karta_block_mask.dtype == torch.bool
    assert processor._global_block_mask[4].item() is True
    assert processor._karta_block_mask[3].item() is True
    assert processor._global_block_mask[2].item() is False


def test_subword_split_bypass_blocked(base_constraints, extended_vocab):
    """Ensures split tokens like 'adm' + 'in' and 'by' + 'pass' cannot bypass filters."""
    processor = PaninianKarakaLogitsProcessor(
        constraints=base_constraints,
        tokenizer_vocab=extended_vocab,
    )

    # Case 1: Active clause where Kartā ("AuthorizedUser") is established,
    # followed by trailing fragment "adm" (ID 5) -> next token "in" (ID 6) must be blocked.
    # Token IDs: [0="AuthorizedUser", 9=" ", 11="executes", 9=" ", 5="adm"]
    input_ids_adm = torch.tensor([[0, 9, 11, 9, 5]], dtype=torch.long)
    scores = torch.full((1, len(extended_vocab)), 5.0, dtype=torch.float)

    modified_adm = processor(input_ids_adm, scores.clone())
    assert modified_adm[0, 6].item() == float("-inf"), "Split token 'adm' + 'in' was not masked"
    assert modified_adm[0, 2].item() == 5.0

    # Case 2: Trailing fragment "by" (ID 7) -> next token "pass" (ID 8) must be globally blocked.
    input_ids_by = torch.tensor([[0, 9, 11, 9, 7]], dtype=torch.long)
    modified_by = processor(input_ids_by, scores.clone())
    assert modified_by[0, 8].item() == float("-inf"), "Split token 'by' + 'pass' was not masked"


def test_clause_boundary_resets_karaka_slots(base_constraints, extended_vocab):
    """Verifies that terminal punctuation resets active Karaka slots for the next clause."""
    processor = PaninianKarakaLogitsProcessor(
        constraints=base_constraints,
        tokenizer_vocab=extended_vocab,
    )

    # Clause 1 establishes Kartā ("AuthorizedUser executes")
    clause_1_ids = torch.tensor([[0, 9, 11]], dtype=torch.long)
    scores = torch.full((1, len(extended_vocab)), 5.0, dtype=torch.float)
    _ = processor(clause_1_ids, scores.clone())

    # Clause 2 begins after terminal period "." (ID 10)
    clause_2_ids = torch.tensor([[0, 9, 11, 10, 9]], dtype=torch.long)
    _ = processor(clause_2_ids, scores.clone())

    assert processor._last_clause_index == 1
    assert processor.active_slots["kartā"] is None
