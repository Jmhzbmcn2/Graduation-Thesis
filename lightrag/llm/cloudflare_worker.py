"""
Cloudflare Worker LLM binding for LightRAG.

This binding calls a custom Cloudflare Worker endpoint that expects:
  Request:  POST / with { prompt, systemPrompt, history: [{role, content}] }
  Response: { response: "..." }

This avoids the OpenAI SDK and calls the Worker directly via aiohttp.
"""

import logging

import pipmaster as pm

if not pm.is_installed("aiohttp"):
    pm.install("aiohttp")

import aiohttp

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import logger


class CloudflareWorkerError(Exception):
    """Error from Cloudflare Worker API"""

    pass


class CloudflareWorkerRetryError(Exception):
    """Retryable error from Cloudflare Worker API"""

    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(CloudflareWorkerRetryError),
)
async def cloudflare_worker_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    base_url: str = "https://text-generation.linhngocut1508.workers.dev/",
    api_key: str = "12345678",
    timeout: int | None = 120,
    **kwargs,
) -> str:
    """Complete a prompt using a custom Cloudflare Worker endpoint.

    Args:
        prompt: The user prompt text.
        system_prompt: Optional system prompt.
        history_messages: Optional list of {role, content} message dicts.
        base_url: The Cloudflare Worker URL.
        api_key: Bearer token for auth.
        timeout: Request timeout in seconds.
        **kwargs: Additional kwargs (ignored, for compatibility).

    Returns:
        The generated text response.
    """
    if history_messages is None:
        history_messages = []

    # Remove kwargs that are not relevant to this binding
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("response_format", None)
    kwargs.pop("stream", None)
    kwargs.pop("openai_client_configs", None)

    # Build the request body matching the Worker's expected format
    payload = {
        "prompt": prompt,
        "systemPrompt": system_prompt or "",
        "history": [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in history_messages
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Ensure base_url ends with /
    if not base_url.endswith("/"):
        base_url += "/"

    logger.debug(f"===== Cloudflare Worker LLM Call =====")
    logger.debug(f"URL: {base_url}")
    logger.debug(f"Prompt length: {len(prompt)}")
    logger.debug(f"History messages: {len(history_messages)}")

    client_timeout = aiohttp.ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                base_url, json=payload, headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("response", "")
                    
                    # Handle cases where the worker nests the CF AI response object
                    if isinstance(content, dict):
                        content = content.get("response", "")
                        
                    if not content or not isinstance(content, str) or not content.strip():
                        logger.warning("Empty response from Cloudflare Worker")
                        raise CloudflareWorkerRetryError(
                            "Empty response from Cloudflare Worker"
                        )
                    logger.debug(f"Response content len: {len(content)}")
                    return content
                elif resp.status in (429, 500, 502, 503, 504):
                    # Retryable errors
                    error_text = await resp.text()
                    logger.warning(
                        f"Cloudflare Worker retryable error {resp.status}: {error_text}"
                    )
                    raise CloudflareWorkerRetryError(
                        f"HTTP {resp.status}: {error_text}"
                    )
                else:
                    error_text = await resp.text()
                    logger.error(
                        f"Cloudflare Worker error {resp.status}: {error_text}"
                    )
                    raise CloudflareWorkerError(
                        f"HTTP {resp.status}: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Cloudflare Worker connection error: {e}")
        raise CloudflareWorkerRetryError(f"Connection error: {e}")
