from __future__ import annotations

import pytest

from lingya.llm.base import ToolDefinition, ToolParameter


class TestToolParameter:
    def test_default_values(self):
        p = ToolParameter(name="query")
        assert p.name == "query"
        assert p.type == "string"
        assert p.description == ""
        assert p.required is True

    def test_custom_values(self):
        p = ToolParameter(
            name="count",
            type="integer",
            description="Number of items",
            required=False,
        )
        assert p.name == "count"
        assert p.type == "integer"
        assert p.description == "Number of items"
        assert p.required is False


class TestToolDefinition:
    def test_no_parameters(self):
        t = ToolDefinition(name="ping", description="Check if alive")
        fmt = t.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "ping"
        assert fmt["function"]["description"] == "Check if alive"
        assert fmt["function"]["parameters"]["properties"] == {}

    def test_with_parameters(self):
        t = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=[
                ToolParameter(name="query", description="Search term"),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                    required=False,
                ),
            ],
        )
        fmt = t.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "search"
        props = fmt["function"]["parameters"]["properties"]
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "Search term"
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["description"] == "Max results"
        required = fmt["function"]["parameters"]["required"]
        assert "query" in required
        assert "limit" not in required

    def test_all_parameters_required(self):
        t = ToolDefinition(
            name="add",
            description="Add numbers",
            parameters=[
                ToolParameter(name="a", type="integer"),
                ToolParameter(name="b", type="integer"),
            ],
        )
        fmt = t.to_openai_format()
        required = fmt["function"]["parameters"]["required"]
        assert required == ["a", "b"]
