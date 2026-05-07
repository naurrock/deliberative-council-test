"""Tests for council.graph — communication graph construction."""

import pytest

from council.graph import build_communication_graph, validate_graph


class TestFullGraph:
    def test_two_agents(self):
        graph = build_communication_graph(["a", "b"], strategy="full")
        assert "a" in graph
        assert "b" in graph
        assert "b" in graph["a"]
        assert "a" in graph["b"]

    def test_three_agents(self):
        agents = ["a", "b", "c"]
        graph = build_communication_graph(agents, strategy="full")
        for agent in agents:
            assert len(graph[agent]) == 2  # Each sees 2 others

    def test_single_agent(self):
        graph = build_communication_graph(["a"], strategy="full")
        assert graph["a"] == []

    def test_empty_agents(self):
        graph = build_communication_graph([], strategy="full")
        assert graph == {}


class TestSparseGraph:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            build_communication_graph(["a", "b"], strategy="sparse")


class TestValidateGraph:
    def test_valid_full_graph(self):
        agents = ["a", "b", "c"]
        graph = build_communication_graph(agents, strategy="full")
        assert validate_graph(graph, agents) is True

    def test_agent_missing_from_graph(self):
        graph = {"a": ["b"], "b": ["a"]}
        assert validate_graph(graph, ["a", "b", "c"]) is False

    def test_agent_sees_itself(self):
        graph = {"a": ["a", "b"], "b": ["a"]}
        assert validate_graph(graph, ["a", "b"]) is False

    def test_agent_sees_unknown(self):
        graph = {"a": ["b", "x"], "b": ["a"]}
        assert validate_graph(graph, ["a", "b"]) is False
