"""Tests for step assignment logic."""

import pytest

from src.pipeline.step_assignment import StepRange, assign_steps


class TestStepAssignment:
    """Test cases for step assignment across ranks."""

    def test_single_rank(self):
        """Single rank should handle all steps."""
        result = assign_steps(total_steps=28, world_size=1, rank=0)
        assert result.start == 0
        assert result.end == 28

    def test_two_ranks_even_split(self):
        """Two ranks should split steps evenly."""
        result0 = assign_steps(total_steps=28, world_size=2, rank=0)
        result1 = assign_steps(total_steps=28, world_size=2, rank=1)

        assert result0.start == 0
        assert result0.end == 14
        assert result1.start == 14
        assert result1.end == 28

    def test_four_ranks(self):
        """Four ranks should each handle 7 steps."""
        for rank in range(4):
            result = assign_steps(total_steps=28, world_size=4, rank=rank)
            assert result.count == 7
            assert result.start == rank * 7

    def test_seven_ranks(self):
        """Seven ranks should each handle 4 steps."""
        for rank in range(7):
            result = assign_steps(total_steps=28, world_size=7, rank=rank)
            assert result.count == 4
            assert result.start == rank * 4

    def test_no_gaps(self):
        """All steps should be covered with no gaps."""
        world_size = 7
        total_steps = 28
        covered = set()

        for rank in range(world_size):
            result = assign_steps(total_steps, world_size, rank)
            for step in range(result.start, result.end):
                assert step not in covered, f"Step {step} covered by multiple ranks"
                covered.add(step)

        assert covered == set(range(total_steps)), "Not all steps covered"

    def test_invalid_total_steps(self):
        """Should raise ValueError for invalid total_steps."""
        with pytest.raises(ValueError):
            assign_steps(total_steps=0, world_size=1, rank=0)
        with pytest.raises(ValueError):
            assign_steps(total_steps=-1, world_size=1, rank=0)

    def test_invalid_world_size(self):
        """Should raise ValueError for invalid world_size."""
        with pytest.raises(ValueError):
            assign_steps(total_steps=28, world_size=0, rank=0)

    def test_invalid_rank(self):
        """Should raise ValueError for invalid rank."""
        with pytest.raises(ValueError):
            assign_steps(total_steps=28, world_size=4, rank=4)
        with pytest.raises(ValueError):
            assign_steps(total_steps=28, world_size=4, rank=-1)

    def test_not_divisible(self):
        """Should raise ValueError when total_steps not divisible by world_size."""
        with pytest.raises(ValueError):
            assign_steps(total_steps=29, world_size=7, rank=0)


class TestStepRange:
    """Test cases for StepRange dataclass."""

    def test_count_property(self):
        """Count should return the number of steps."""
        sr = StepRange(start=0, end=7)
        assert sr.count == 7

    def test_iteration(self):
        """Should be iterable."""
        sr = StepRange(start=5, end=10)
        assert list(sr) == [5, 6, 7, 8, 9]

    def test_invalid_range(self):
        """Should raise ValueError for invalid ranges."""
        with pytest.raises(ValueError):
            StepRange(start=-1, end=5)
        with pytest.raises(ValueError):
            StepRange(start=10, end=5)
