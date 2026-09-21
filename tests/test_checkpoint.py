"""Check that numeric model weights retain schema-specific calibration."""

import torch

from tacit import Boolean, Choice, EncoderConfig, SignalConfig, SignalTacit


def test_signal_checkpoint_roundtrip_and_partial_measurement_retention(tmp_path):
    torch.manual_seed(2)
    config = SignalConfig(("load", "errors"), EncoderConfig(d_model=32, n_layers=1))
    questions = {"alert": Boolean("alert"), "route": Choice("route", ["stay", "retry"])}
    original = SignalTacit(config, questions).eval()
    original.temperatures[0].temperature.fill_(2.0)
    original.temperatures[1].temperature.fill_(0.5)
    path = tmp_path / "model.pt"
    torch.save(original.state_dict(), path)
    loaded = SignalTacit(config, questions).eval()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    values = torch.tensor([[[0.2, 0.7]]])
    with torch.no_grad():
        expected, _ = original(values, calibrated=True)
        actual, state = loaded(values, calibrated=True)
        for name in questions:
            torch.testing.assert_close(expected[name], actual[name])
        storage = state.nbytes
        # Long unrelated updates must not alter the explicit errors register.
        for _ in range(10):
            updates = torch.zeros(1, 100, 2)
            mask = torch.zeros_like(updates, dtype=torch.bool)
            mask[..., 0] = True
            _, state = loaded(updates, mask, state)
        assert state.pos == 1001
        torch.testing.assert_close(state.values, torch.tensor([[0.0, 0.7]]))
        assert state.nbytes == storage
        for pair in state.blocks:
            for tensor in pair:
                if tensor is not None:
                    assert (
                        tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
                    )
