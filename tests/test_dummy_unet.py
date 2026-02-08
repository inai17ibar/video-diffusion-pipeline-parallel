"""Tests for DummyUNet model."""

import torch

from src.models.dummy_unet import DummyUNet


class TestDummyUNet:
    """Test cases for DummyUNet."""

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        model = DummyUNet(channels=8)
        x = torch.randn(1, 8, 8, 32, 32)
        out = model(x, step=0)
        assert out.shape == x.shape

    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        model = DummyUNet(channels=4)
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 4, 8, 32, 32)
            out = model(x, step=0)
            assert out.shape == x.shape

    def test_different_resolutions(self):
        """Should work with different spatial resolutions."""
        model = DummyUNet(channels=8)
        for h, w in [(16, 16), (32, 32), (64, 64)]:
            x = torch.randn(1, 8, 8, h, w)
            out = model(x, step=0)
            assert out.shape == x.shape

    def test_step_parameter_accepted(self):
        """Model should accept step parameter."""
        model = DummyUNet(channels=8)
        x = torch.randn(1, 8, 8, 32, 32)
        # Should not raise
        for step in [0, 10, 27]:
            out = model(x, step=step)
            assert out is not None
