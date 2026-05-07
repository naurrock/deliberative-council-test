"""
Tool registry and Jina.ai integration for Deliberative Council.

Provides search and content extraction via Jina.ai, plus a tool registry
for managing available tools for agents.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from council.config import ResearchConfig
from council.types import EvidenceSource

logger = logging.getLogger(__name__)


# ── Jina.ai Client ─────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search result from Jina.ai."""

    url: str
    title: str
    snippet: str
    rank: int = 0


@dataclass
class ExtractResult:
    """Result of content extraction from a URL."""

    url: str
    title: str
    content: str
    success: bool = True
    error: str | None = None


class JinaClient:
    """Client for Jina.ai search and extraction APIs.

    Uses s.jina.ai for search and r.jina.ai for content extraction.
    No API key required for basic usage.
    """

    def __init__(self, config: ResearchConfig | None = None):
        self.config = config or ResearchConfig()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Accept": "text/plain"},
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def search(
        self, query: str, max_results: int | None = None
    ) -> list[SearchResult]:
        """Search using Jina.ai s.jina.ai endpoint.

        Returns a list of search results with URL, title, and snippet.
        """
        max_res = max_results or self.config.max_search_results
        search_url = f"{self.config.jina_search_url}/{query}"

        try:
            session = await self._get_session()
            async with session.get(
                search_url,
                params={"count": str(max_res)},
            ) as response:
                if response.status != 200:
                    logger.warning(
                        f"Jina search returned status {response.status} for query: {query}"
                    )
                    return []

                text = await response.text()
                return self._parse_search_results(text, query)

        except asyncio.TimeoutError:
            logger.warning(f"Jina search timed out for query: {query}")
            return []
        except Exception as e:
            logger.warning(f"Jina search failed for query '{query}': {e}")
            return []

    async def extract(self, url: str) -> ExtractResult:
        """Extract content from a URL using Jina.ai r.jina.ai endpoint.

        Returns the extracted text content from the page.
        """
        extract_url = f"{self.config.jina_extract_url}/{url}"

        try:
            session = await self._get_session()
            async with session.get(extract_url) as response:
                if response.status != 200:
                    return ExtractResult(
                        url=url,
                        title="",
                        content="",
                        success=False,
                        error=f"HTTP {response.status}",
                    )

                text = await response.text()
                title = self._extract_title(text, url)
                return ExtractResult(
                    url=url,
                    title=title,
                    content=text[:10_000],  # Cap at 10K chars
                    success=True,
                )

        except asyncio.TimeoutError:
            return ExtractResult(
                url=url, title="", content="", success=False, error="Timeout"
            )
        except Exception as e:
            return ExtractResult(
                url=url, title="", content="", success=False, error=str(e)[:200]
            )

    def _parse_search_results(self, text: str, query: str) -> list[SearchResult]:
        """Parse Jina.ai search response into structured results.

        Jina.ai returns plain text with URL-title pairs. We parse them
        into SearchResult objects.
        """
        results = []
        seen_urls = set()

        # Jina search results typically appear as numbered lists
        # with URLs and descriptions
        url_pattern = re.compile(r'https?://[^\s<>"\')\]]+')
        urls = url_pattern.findall(text)

        for i, url in enumerate(urls[: self.config.max_search_results]):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Try to extract a snippet from text around the URL
            url_pos = text.find(url)
            snippet_start = max(0, url_pos - 100)
            snippet_end = min(len(text), url_pos + len(url) + 200)
            snippet = text[snippet_start:snippet_end].replace(url, "").strip()
            # Clean up snippet
            snippet = re.sub(r'\s+', ' ', snippet)[:300]

            results.append(
                SearchResult(
                    url=url,
                    title=self._extract_title_from_url(url),
                    snippet=snippet,
                    rank=i + 1,
                )
            )

        return results

    @staticmethod
    def _extract_title(text: str, fallback_url: str) -> str:
        """Extract a title from HTML content or use fallback."""
        # Look for <title> tag
        title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            if title:
                return title[:200]
        return JinaClient._extract_title_from_url(fallback_url)

    @staticmethod
    def _extract_title_from_url(url: str) -> str:
        """Generate a title from a URL as fallback."""
        # Remove protocol and path, use domain
        domain_match = re.search(r'://([^/]+)', url)
        if domain_match:
            return domain_match.group(1)
        return url[:100]


# ── Tool Registry ──────────────────────────────────────────────────────


@dataclass
class ToolSpec:
    """Specification for a tool available to agents."""

    name: str
    description: str
    function: Any  # The actual callable
    requires_async: bool = False


class ToolRegistry:
    """Registry of tools available to research and scout agents.

    Tools are callables that agents can invoke during their work.
    The registry provides discovery and execution.
    """

    def __init__(self, jina_client: JinaClient | None = None):
        self._tools: dict[str, ToolSpec] = {}
        self.jina = jina_client or JinaClient()

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register the standard set of built-in tools."""
        self.register(
            ToolSpec(
                name="web_search",
                description="Search the web using Jina.ai. Returns URLs, titles, and snippets.",
                function=self.jina.search,
                requires_async=True,
            )
        )
        self.register(
            ToolSpec(
                name="extract_content",
                description="Extract clean text content from a URL using Jina.ai.",
                function=self.jina.extract,
                requires_async=True,
            )
        )

    def register(self, tool: ToolSpec) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def all_tools(self) -> list[ToolSpec]:
        """Return all registered tools."""
        return list(self._tools.values())

    def tool_descriptions(self) -> str:
        """Return a formatted description of all tools for prompt injection."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    async def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with the given arguments."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        if tool.requires_async:
            return await tool.function(**kwargs)
        else:
            return tool.function(**kwargs)

    async def close(self) -> None:
        """Close underlying resources."""
        await self.jina.close()
