# [MODULE DOC]
# Environment-based provider configuration for the tau_ai layer. Defines
# default endpoints/timeouts/retry settings, the cache retention preference
# type and constants, the runtime auth value and resolver/observer callable
# types, frozen config dataclasses for OpenAI-compatible and Anthropic
# providers, and a loader that builds an OpenAI-compatible config from
# environment variables (with private parsing helpers).
"""Environment-based provider configuration helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from os import environ
from typing import Literal

from tau_agent.types import JSONValue

DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES = 2
DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS = 1.0

# Prompt-cache retention preferences. "short" uses the provider default TTL
# (5 minutes on Anthropic), "long" requests the 1 hour TTL, and "none" disables
# cache breakpoints entirely for backends that reject them.
type CacheRetention = Literal["none", "short", "long"]

CACHE_RETENTION_NONE: CacheRetention = "none"
CACHE_RETENTION_SHORT: CacheRetention = "short"
CACHE_RETENTION_LONG: CacheRetention = "long"


@dataclass(frozen=True, slots=True)
class RuntimeProviderAuth:
    # [CLASS DOC]
    # Frozen dataclass of request auth resolved right before a provider call: API
    # key, optional base URL override, optional extra headers.
    """Request auth resolved immediately before a provider call."""

    api_key: str
    base_url: str | None = None
    headers: Mapping[str, str] | None = None


type RuntimeProviderAuthResolver = Callable[[], Awaitable[RuntimeProviderAuth]]
type RuntimeResponseHeadersObserver = Callable[[Mapping[str, str]], None]


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    # [CLASS DOC]
    # Frozen dataclass configuring an OpenAI-compatible endpoint: API key (hidden
    # from repr), base URL, headers (hidden from repr), timeout, retry count and
    # max retry delay, wire API name, max tokens, image support, reasoning effort
    # and its parameter name, thinking format, compat flags, model aliases,
    # whether to send reasoning effort none, display provider name, optional
    # response provider header, whether to omit the Authorization header, optional
    # runtime credential resolver and response-headers observer, and whether to
    # infer the wire API from the model id.
    """Configuration for an OpenAI-compatible chat completions endpoint."""

    api_key: str = field(repr=False)
    base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    headers: Mapping[str, str] | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    api: str = "openai-completions"
    max_tokens: int | None = None
    supports_images: bool = False
    reasoning_effort: str | None = None
    reasoning_effort_parameter: str = "reasoning_effort"
    thinking_format: str = "openai"
    compat: Mapping[str, JSONValue] = field(default_factory=dict)
    model_aliases: Mapping[str, str] = field(default_factory=dict)
    include_reasoning_effort_none: bool = False
    provider_name: str = "OpenAI-compatible provider"
    response_provider_header: str | None = None
    omit_authorization_header: bool = False
    credential_resolver: RuntimeProviderAuthResolver | None = None
    response_headers_observer: RuntimeResponseHeadersObserver | None = None
    infer_api_from_model: bool = True


@dataclass(frozen=True, slots=True)
class AnthropicConfig:
    # [CLASS DOC]
    # Frozen dataclass configuring Anthropic's Messages API: API key, whether to
    # use bearer (OAuth) auth, base URL, headers, timeout, retries and max retry
    # delay, max tokens, image support, thinking budget/effort/mode, provider
    # name, optional OAuth system prompt, cache retention preference, whether to
    # put cache control on tools, and optional runtime credential resolver.
    """Configuration for Anthropic's Messages API."""

    api_key: str
    bearer_auth: bool = False
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    max_tokens: int | None = None
    supports_images: bool = False
    thinking_budget_tokens: int | None = None
    thinking_effort: str | None = None
    thinking_mode: str = "budget"
    provider_name: str = "Anthropic"
    oauth_system_prompt: str | None = None
    cache_retention: CacheRetention = CACHE_RETENTION_SHORT
    cache_control_on_tools: bool = True
    credential_resolver: RuntimeProviderAuthResolver | None = None


def openai_compatible_config_from_env(
    *,
    api_key_var: str = "OPENAI_API_KEY",
    base_url_var: str = "OPENAI_BASE_URL",
    timeout_seconds_var: str = "OPENAI_TIMEOUT_SECONDS",
    max_retries_var: str = "OPENAI_MAX_RETRIES",
    max_retry_delay_seconds_var: str = "OPENAI_MAX_RETRY_DELAY_SECONDS",
    default_timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    default_max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    default_max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
) -> OpenAICompatibleConfig:
    # [FUNCTION DOC]
    # Build an OpenAICompatibleConfig from environment variables. Inputs: names
    # of the API key, base URL, timeout, max retries and max retry delay variables,
    # plus defaults for the numeric values. Reads the API key (required), base URL
    # (default OpenAI URL, trailing slashes stripped) and parses the numeric values
    # via the helpers. Output: the config (other fields default).
    # Raises RuntimeError when the API key is missing/empty or a numeric variable
    # is invalid. Side effects: reads process environment.
    """Load OpenAI-compatible provider configuration from environment variables."""
    api_key = environ.get(api_key_var)
    if not api_key:
        msg = f"Missing required environment variable: {api_key_var}"
        raise RuntimeError(msg)

    timeout_seconds = _timeout_seconds_from_env(timeout_seconds_var, default_timeout_seconds)
    max_retries = _non_negative_int_from_env(max_retries_var, default_max_retries)
    max_retry_delay_seconds = _non_negative_float_from_env(
        max_retry_delay_seconds_var, default_max_retry_delay_seconds
    )
    return OpenAICompatibleConfig(
        api_key=api_key,
        base_url=environ.get(base_url_var, DEFAULT_OPENAI_COMPATIBLE_BASE_URL).rstrip("/"),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_retry_delay_seconds=max_retry_delay_seconds,
    )


def _timeout_seconds_from_env(name: str, default: float) -> float:
    # [FUNCTION DOC]
    # HELPER: read a positive float from the named env var, returning the default
    # when unset. Raises RuntimeError if not a number or not greater than 0.
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        timeout_seconds = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable must be a number: {name}") from exc
    if timeout_seconds <= 0:
        raise RuntimeError(f"Environment variable must be greater than 0: {name}")
    return timeout_seconds


def _non_negative_int_from_env(name: str, default: int) -> int:
    # [FUNCTION DOC]
    # HELPER: read a non-negative int from the named env var, returning the
    # default when unset. Raises RuntimeError if not an integer or negative.
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable must be an integer: {name}") from exc
    if value < 0:
        raise RuntimeError(f"Environment variable must be 0 or greater: {name}")
    return value


def _non_negative_float_from_env(name: str, default: float) -> float:
    # [FUNCTION DOC]
    # HELPER: read a non-negative float from the named env var, returning the
    # default when unset. Raises RuntimeError if not a number or negative.
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable must be a number: {name}") from exc
    if value < 0:
        raise RuntimeError(f"Environment variable must be 0 or greater: {name}")
    return value
