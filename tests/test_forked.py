"""The fork graph must not leak candidate order or neighboring examples."""

import pytest
import torch

from tacit.forked import ForkedTacit, fork_masks, pack_fork_batch, pack_forks


def test_fork_visibility():
    owners = torch.tensor([[0, 0, 1, 1, 2, 2, -1]])
    positions = torch.tensor([[0, 1, 2, 3, 2, 3, 0]])
    mask = fork_masks(owners, positions, 128)["full_attention"][0, 0]
    assert mask[:2, :2].all() and not mask[:2, 2:].any()
    assert mask[2:4, :4].all() and not mask[2:4, 4:].any()
    assert mask[4:6, :2].all() and mask[4:6, 4:6].all() and not mask[4:6, 2:4].any()
    assert mask[-1, -1] and not mask[:-1, -1].any()


def test_fork_encoder_equivariance_and_gradients():
    pytest.importorskip("transformers")
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(1)
    config = ModernBertConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        local_attention=16,
        global_attn_every_n_layers=2,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = ForkedTacit(ModernBertModel(config)).eval()
    states, branches = [[1, 2, 3, 4]], [[5, 7, 8], [5, 9], [5, 10, 11, 12]]
    original = model(**pack_forks(states, branches, 0))
    order = [2, 0, 1]
    reordered = model(**pack_forks(states, [branches[i] for i in order], 0))
    torch.testing.assert_close(reordered, original[:, order], atol=1e-6, rtol=1e-5)
    extended = model(**pack_forks(states, branches + [[5, 18, 19]], 0))
    torch.testing.assert_close(extended[:, :3], original, atol=1e-6, rtol=1e-5)
    batch = model(**pack_forks([states[0], [1, 9, 9, 8, 7]], branches, 0))
    torch.testing.assert_close(batch[:1], original, atol=1e-6, rtol=1e-5)
    original.square().mean().backward()
    grad = model.encoder.get_input_embeddings().weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_different_questions_and_option_counts_do_not_cross_contaminate():
    pytest.importorskip("transformers")
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(7)
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        pad_token_id=0,
        local_attention=16,
        global_attn_every_n_layers=2,
    )
    config._attn_implementation = "sdpa"
    model = ForkedTacit(ModernBertModel(config)).eval()
    states = [[1, 2, 3], [1, 4, 5, 6, 7]]
    schemas = [[[8, 9], [8, 10]], [[8, 11], [8, 12, 13], [8, 14]]]
    batch = model(**pack_fork_batch(states, schemas, 0))
    for i in range(2):
        single = model(**pack_forks([states[i]], schemas[i], 0))
        torch.testing.assert_close(batch[i, : len(schemas[i])], single[0], atol=1e-6, rtol=1e-5)
    assert torch.isneginf(batch[0, 2])
    # Exclude padded candidate logits from finite-logit objectives.
    loss = batch[0, :2].square().mean() + batch[1, :3].square().mean()
    loss.backward()
    assert torch.isfinite(model.readout[-1].weight.grad).all()
    with pytest.raises(ValueError):
        pack_fork_batch(states, schemas[:1], 0)
