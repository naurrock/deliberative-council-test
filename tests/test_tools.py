"""Comprehensive tests for council.tools — tool registry and Jina.ai client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from council.config import ResearchConfig
from council.tools import (
    ExtractResult,
    JinaClient,
    SearchResult,
    ToolRegistry,
    ToolSpec,
)


# ── SearchResult Tests ─────────────────────────────────────────────────


class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(
            url="https://example.com",
            title="Example",
            snippet="A snippet",
            rank=1,
        )
        assert sr.url == "https://example.com"
        assert sr.rank == 1


# ── ExtractResult Tests ────────────────────────────────────────────────


class TestExtractResult:
    def test_successful_result(self):
        er = ExtractResult(
            url="https://example.com",
            title="Example Page",
            content="Extracted content here",
            success=True,
        )
        assert er.success is True
        assert er.error is None

    def test_failed_result(self):
        er = ExtractResult(
            url="https://example.com",
            title="",
            content="",
            success=False,
            error="HTTP 404",
        )
        assert er.success is False
        assert er.error == "HTTP 404"


# ── JinaClient Tests ──────────────────────────────────────────────────


class TestJinaClient:
    def test_default_config(self):
        client = JinaClient()
        assert "jina.ai" in client.config.jina_search_url

    def test_custom_config(self):
        config = ResearchConfig(
            jina_search_url="https://custom.search.api",
            jina_extract_url="https://custom.extract.api",
        )
        client = JinaClient(config)
        assert client.config.jina_search_url == "https://custom.search.api"

    @pytest.mark.asyncio
    async def test_search_with_mock(self):
        """Test search with mocked HTTP response."""
        client = JinaClient()
        mock_text = """
        1. https://example.com/quantum
        Quantum computing uses quantum mechanics to process information.
        2. https://example.com/ai
        AI is transforming industries.
        """

        with patch.object(client, '_get_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_text)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            session = AsyncMock()
            session.get = MagicMock(return_value=mock_response)
            mock_session.return_value = session

            results = await client.search("quantum computing")
            assert len(results) >= 1
            assert results[0].url == "https://example.com/quantum"

    @pytest.mark.asyncio
    async def test_search_failure_returns_empty(self):
        """Failed search should return empty list."""
        client = JinaClient()

        with patch.object(client, '_get_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            session = AsyncMock()
            session.get = MagicMock(return_value=mock_response)
            mock_session.return_value = session

            results = await client.search("test query")
            assert results == []

    @pytest.mark.asyncio
    async def test_extract_with_mock(self):
        """Test content extraction with mocked HTTP response."""
        client = JinaClient()
        mock_content = "This is extracted content from the page."

        with patch.object(client, '_get_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_content)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            session = AsyncMock()
            session.get = MagicMock(return_value=mock_response)
            mock_session.return_value = session

            result = await client.extract("https://example.com/article")
            assert result.success is True
            assert "extracted content" in result.content

    @pytest.mark.asyncio
    async def test_extract_failure(self):
        """Failed extraction should return error result."""
        client = JinaClient()

        with patch.object(client, '_get_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            session = AsyncMock()
            session.get = MagicMock(return_value=mock_response)
            mock_session.return_value = session

            result = await client.extract("https://example.com/missing")
            assert result.success is False
            assert "HTTP 404" in result.error

    @pytest.mark.asyncio
    async def test_close(self):
        """Closing client should close the session."""
        client = JinaClient()
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._session = mock_session

        await client.close()
        mock_session.close.assert_called_once()
        assert client._session is None  # Session should be set to None after close

    @pytest.mark.asyncio
    async def test_close_when_no_session(self):
        """Closing without a session should not error."""
        client = JinaClient()
        await client.close()  # Should not raise


# ── JinaClient Parsing Tests ──────────────────────────────────────────


class TestJinaClientParsing:
    def test_parse_search_results_extracts_urls(self):
        """Should extract URLs from search response text."""
        client = JinaClient()
        text = "Visit https://example.com/article for more info. Also see https://other.com/page"
        results = client._parse_search_results(text, "test")
        assert len(results) >= 1
        assert any("example.com" in r.url for r in results)

    def test_parse_search_results_deduplicates_urls(self):
        """Should not include duplicate URLs."""
        client = JinaClient()
        text = "https://example.com https://example.com https://other.com"
        results = client._parse_search_results(text, "test")
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))

    def test_extract_title_from_url(self):
        """Should extract domain as title from URL."""
        title = JinaClient._extract_title_from_url("https://www.example.com/page/article")
        assert "example.com" in title

    def test_extract_title_from_html(self):
        """Should extract title from HTML content."""
        html = "<html><head><title>My Page Title</title></head><body>Content</body></html>"
        title = JinaClient._extract_title(html, "https://example.com")
        assert title == "My Page Title"


# ── ToolRegistry Tests ────────────────────────────────────────────────


class TestToolRegistry:
    def test_builtin_tools_registered(self):
        """Should have web_search and extract_content tools by default."""
        registry = ToolRegistry()
        tools = registry.all_tools()
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "extract_content" in names

    def test_tool_descriptions(self):
        """Should return formatted tool descriptions."""
        registry = ToolRegistry()
        desc = registry.tool_descriptions()
        assert "web_search" in desc
        assert "extract_content" in desc

    def test_get_existing_tool(self):
        """Should return a tool by name."""
        registry = ToolRegistry()
        tool = registry.get("web_search")
        assert tool is not None
        assert tool.name == "web_search"

    def test_get_nonexistent_tool(self):
        """Should return None for unknown tools."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_register_custom_tool(self):
        """Should register custom tools."""
        registry = ToolRegistry()
        custom = ToolSpec(
            name="custom_tool",
            description="A custom tool for testing",
            function=lambda x: x * 2,
        )
        registry.register(custom)
        tool = registry.get("custom_tool")
        assert tool is not None
        assert tool.function(5) == 10

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self):
        """Executing an unknown tool should raise ValueError."""
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            await registry.execute("nonexistent_tool")

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        """Should execute synchronous tools."""
        registry = ToolRegistry()
        custom = ToolSpec(
            name="sync_tool",
            description="Sync tool",
            function=lambda x: x + 1,
            requires_async=False,
        )
        registry.register(custom)
        result = await registry.execute("sync_tool", x=5)
        assert result == 6

    @pytest.mark.asyncio
    async def test_close_closes_jina(self):
        """Closing registry should close Jina client."""
        registry = ToolRegistry()
        with patch.object(registry.jina, 'close', new_callable=AsyncMock) as mock_close:
            await registry.close()
            mock_close.assert_called_once()
