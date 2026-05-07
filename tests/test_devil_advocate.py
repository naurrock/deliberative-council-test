"""Tests for council.devil_advocate — devil's advocate assignment."""

import pytest

from council.devil_advocate import assign_devils_advocate


class TestNoneStrategy:
    def test_returns_empty_assignments(self):
        result = assign_devils_advocate(["a", "b", "c"], round_num=0, strategy="none")
        assert len(result) == 3
        for agent_id, assignment in result.items():
            assert assignment.devil_agent_id is None

    def test_round_number_irrelevant(self):
        r1 = assign_devils_advocate(["a", "b"], round_num=0, strategy="none")
        r2 = assign_devils_advocate(["a", "b"], round_num=5, strategy="none")
        assert r1.keys() == r2.keys()


class TestExtensionStrategies:
    def test_rotate_not_implemented(self):
        with pytest.raises(NotImplementedError):
            assign_devils_advocate(["a", "b"], round_num=0, strategy="rotate")

    def test_weakest_not_implemented(self):
        with pytest.raises(NotImplementedError):
            assign_devils_advocate(["a", "b"], round_num=0, strategy="weakest")

    def test_random_not_implemented(self):
        with pytest.raises(NotImplementedError):
            assign_devils_advocate(["a", "b"], round_num=0, strategy="random")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            assign_devils_advocate(["a", "b"], round_num=0, strategy="invalid")
