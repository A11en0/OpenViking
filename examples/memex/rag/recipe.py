"""
Memex RAG Recipe - RAG flow implementation for Memex.

Based on examples/common/recipe.py pattern.
"""

from typing import Any, Optional

from openai import OpenAI

from client import MemexClient
from config import MemexConfig


DEFAULT_SYSTEM_PROMPT = """You are Memex, a personal knowledge assistant. 
You help users find and understand information from their personal knowledge base.

When answering questions:
1. Base your answers on the provided context from the knowledge base
2. If the context doesn't contain relevant information, say so clearly
3. Cite sources when possible by mentioning the document or section
4. Be concise but thorough

Context from knowledge base:
{context}
"""


class MemexRecipe:
    """RAG recipe for Memex - handles search, context building, and LLM generation."""

    def __init__(
        self,
        client: MemexClient,
        config: Optional[MemexConfig] = None,
    ):
        """Initialize the RAG recipe.

        Args:
            client: Memex client instance.
            config: Memex configuration.
        """
        self.client = client
        self.config = config or client.config
        self._llm_client: Optional[OpenAI] = None
        self._chat_history: list[dict[str, str]] = []

    @property
    def llm_client(self) -> OpenAI:
        """Get or create LLM client."""
        if self._llm_client is None:
            if self.config.llm_backend == "openai":
                self._llm_client = OpenAI()
            elif self.config.llm_backend == "volcengine":
                # Volcengine uses OpenAI-compatible API
                import os

                self._llm_client = OpenAI(
                    api_key=os.getenv("ARK_API_KEY"),
                    base_url="https://ark.cn-beijing.volces.com/api/v3",
                )
            else:
                raise ValueError(f"Unsupported LLM backend: {self.config.llm_backend}")
        return self._llm_client

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_uri: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Search the knowledge base.

        Args:
            query: Search query.
            top_k: Number of results.
            target_uri: URI to search in.
            score_threshold: Minimum score threshold.

        Returns:
            List of search results with uri, score, content.
        """
        top_k = top_k or self.config.search_top_k
        target_uri = target_uri or self.config.default_resource_uri
        score_threshold = score_threshold or self.config.search_score_threshold

        results = self.client.search(
            query=query,
            target_uri=target_uri,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # Convert results to list of dicts
        search_results = []
        if hasattr(results, "resources"):
            for r in results.resources:
                result = {
                    "uri": r.uri if hasattr(r, "uri") else str(r),
                    "score": r.score if hasattr(r, "score") else 0.0,
                    "content": r.content if hasattr(r, "content") else "",
                }
                search_results.append(result)
        elif isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    search_results.append(r)
                else:
                    search_results.append({"uri": str(r), "score": 0.0, "content": ""})

        return search_results

    def build_context(self, search_results: list[dict[str, Any]]) -> str:
        """Build context string from search results.

        Args:
            search_results: List of search results.

        Returns:
            Formatted context string.
        """
        if not search_results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, result in enumerate(search_results, 1):
            uri = result.get("uri", "unknown")
            content = result.get("content", "")
            score = result.get("score", 0.0)

            # If content is empty, try to get abstract
            if not content:
                try:
                    content = self.client.abstract(uri)
                except Exception:
                    content = f"[Content from {uri}]"

            context_parts.append(f"[Source {i}] {uri} (relevance: {score:.2f})\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the LLM with messages.

        Args:
            messages: List of message dicts with role and content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated response.
        """
        temperature = temperature if temperature is not None else self.config.llm_temperature
        max_tokens = max_tokens or self.config.llm_max_tokens

        response = self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

    def query(
        self,
        user_query: str,
        search_top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        score_threshold: Optional[float] = None,
        target_uri: Optional[str] = None,
        use_chat_history: bool = False,
    ) -> str:
        """Complete RAG query flow: search -> build context -> generate.

        Args:
            user_query: User's question.
            search_top_k: Number of search results.
            temperature: LLM temperature.
            max_tokens: Max tokens for response.
            system_prompt: Custom system prompt (use {context} placeholder).
            score_threshold: Minimum search score.
            target_uri: URI to search in.
            use_chat_history: Whether to include chat history.

        Returns:
            Generated response.
        """
        # Search knowledge base
        search_results = self.search(
            query=user_query,
            top_k=search_top_k,
            target_uri=target_uri,
            score_threshold=score_threshold,
        )

        # Build context
        context = self.build_context(search_results)

        # Build system prompt
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        formatted_system_prompt = system_prompt.format(context=context)

        # Build messages
        messages = [{"role": "system", "content": formatted_system_prompt}]

        # Add chat history if enabled
        if use_chat_history and self._chat_history:
            messages.extend(self._chat_history)

        # Add current query
        messages.append({"role": "user", "content": user_query})

        # Generate response
        response = self.call_llm(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Update chat history
        if use_chat_history:
            self._chat_history.append({"role": "user", "content": user_query})
            self._chat_history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self) -> None:
        """Clear chat history."""
        self._chat_history = []

    @property
    def chat_history(self) -> list[dict[str, str]]:
        """Get current chat history."""
        return self._chat_history.copy()
