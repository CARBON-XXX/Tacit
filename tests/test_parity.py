"""The parallel and streaming paths must compute the same thing.

If they drift, a model trains fine and then behaves differently the moment it
runs on its own carried state. These tests exist so that failure is loud.
"""

from __future__ import annotations

import pytest
import torch

from tacit import ByteEncoder, EncoderConfig, ResonantBlock, Tacit, TacitConfig, encode_bytes

TOLERANCE = 1e-5


def stream(block: ResonantBlock, x: torch.Tensor) -> torch.Tensor:
    state = block.init_state(x.shape[0], x.device)
    outputs = []
    for t in range(x.shape[1]):
        out, state = block.step(x[:, t : t + 1], state, t)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"phase_selective": False},
        {"mag_selective": True},
        {"conv_kernel": 0},
        {"collapse_every": 0},
        {"collapse_every": 1},
        {"selective_readout": False},
        {"output_gate": False},
        {"n_bands": 8, "n_phases": 8},
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "defaults",
)
def test_block_parallel_matches_streaming(kwargs):
    torch.manual_seed(0)
    config = {"n_bands": 4, "collapse_every": 8, "conv_kernel": 4, **kwargs}
    block = ResonantBlock(64, **config).eval()
    x = torch.randn(2, 24, 64)

    with torch.no_grad():
        parallel = block(x)
        streamed = stream(block, x)

    assert parallel.shape == streamed.shape == x.shape
    assert torch.allclose(parallel, streamed, atol=TOLERANCE)


def test_forward_state_resumes_in_streaming():
    """A state handed back by `forward` must let `step` carry on seamlessly."""
    torch.manual_seed(0)
    block = ResonantBlock(64, n_bands=4, collapse_every=8).eval()
    x = torch.randn(2, 24, 64)

    with torch.no_grad():
        full = block(x)
        prefix_out, state = block(x[:, :16], return_state=True)
        assert torch.allclose(prefix_out, full[:, :16], atol=TOLERANCE)

        rest = []
        for t in range(16, 24):
            out, state = block.step(x[:, t : t + 1], state, t)
            rest.append(out)

    assert torch.allclose(torch.cat(rest, dim=1), full[:, 16:], atol=TOLERANCE)


def test_return_state_rejects_ragged_length():
    block = ResonantBlock(64, n_bands=4, collapse_every=8)
    with pytest.raises(ValueError, match="multiple of the chunk length"):
        block(torch.randn(1, 20, 64), return_state=True)


def test_encoder_parallel_matches_streaming():
    torch.manual_seed(0)
    encoder = ByteEncoder(
        EncoderConfig(d_model=64, n_layers=2, collapse_every=8, register_states=10)
    ).eval()
    tokens = torch.tensor([encode_bytes("payouts have been failing")[:16]])

    with torch.no_grad():
        parallel = encoder(tokens)
        state = encoder.init_state(1, tokens.device)
        streamed, _ = encoder.absorb(tokens, state)

    assert torch.allclose(parallel[:, -1], streamed, atol=TOLERANCE)


def test_padding_does_not_leak_across_a_batch():
    """A short string must encode the same whether or not it shares a batch."""
    torch.manual_seed(0)
    model = Tacit(TacitConfig(encoder=EncoderConfig(d_model=64, n_layers=2))).eval()
    texts = ["short", "a much longer ticket about payouts failing for three days"]

    with torch.no_grad():
        together = model.encode_texts(texts)
        apart = torch.cat([model.encode_texts([t]) for t in texts])

    assert torch.allclose(together, apart, atol=TOLERANCE)
