# [MODULE DOC]
# Anthropic Claude Pro/Max subscription OAuth provider for tau_coding.
# Implements the authorization-code + PKCE browser login (with a local
# callback server on port 53692 and manual paste fallback), token refresh,
# safe error-detail extraction from the token endpoint, and the registered
# provider object that converts stored credentials into runtime bearer auth
# headers. Side effects: opens browsers, runs a local HTTP server, performs
# network requests to Anthropic's token endpoint.
"""Anthropic Claude Pro/Max OAuth provider."""

from __future__ import annotations

import asyncio
import time
import webbrowser
from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from urllib.parse import urlencode

import httpx

from tau_ai.http import create_async_client
from tau_coding.credentials import OAuthCredential
from tau_coding.oauth import (
    OAuthError,
    _start_local_oauth_server,
    create_pkce_pair,
    oauth_credential_is_expired,
    parse_authorization_input,
)
from tau_coding.oauth_types import (
    OAuthAuthInfo,
    OAuthFlowKind,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthRuntimeAuth,
)

ANTHROPIC_OAUTH_PROVIDER = "anthropic"
ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
ANTHROPIC_REDIRECT_URI = "http://localhost:53692/callback"
ANTHROPIC_SCOPE = (
    "org:create_api_key user:profile user:inference user:sessions:claude_code "
    "user:mcp_servers user:file_upload"
)
ANTHROPIC_CALLBACK_PORT = 53692
ANTHROPIC_TOKEN_SKEW_MS = 5 * 60 * 1000
# Token-request fields worth scrubbing out of anything the server echoes back.
_SECRET_REQUEST_FIELDS = ("refresh_token", "code", "code_verifier")


async def login_anthropic(
    *,
    on_auth: Callable[[OAuthAuthInfo], None],
    on_prompt: Callable[[OAuthPrompt], Awaitable[str]],
    on_manual_code_input: Callable[[], Awaitable[str]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    open_browser: bool = True,
    client: httpx.AsyncClient | None = None,
) -> OAuthCredential:
    # [FUNCTION DOC]
    # Run the full Anthropic OAuth login. Inputs (keyword-only): callback to show
    # auth info (URL + instructions), async prompt callback, optional async manual
    # code-input callback, optional progress callback, whether to open a browser,
    # optional HTTP client. Creates a PKCE pair (the verifier doubles as OAuth
    # state), builds the authorize URL, starts the local callback server, reports
    # the URL, optionally opens the browser, then waits for whichever comes first
    # of the server callback or manual input; if neither yields a value, prompts
    # the user to paste the code or redirect URL. Parses the input, checks state,
    # and exchanges the code for tokens.
    # Output: an OAuth credential (access, refresh, expiry ms).
    # Raises OAuthError on state mismatch, missing code, or token failures.
    # Always closes the local server.
    """Run Anthropic's authorization-code + PKCE login flow."""
    verifier, challenge = create_pkce_pair()
    params = {
        "code": "true",
        "client_id": ANTHROPIC_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": ANTHROPIC_REDIRECT_URI,
        "scope": ANTHROPIC_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    url = f"{ANTHROPIC_AUTHORIZE_URL}?{urlencode(params)}"
    server = await _start_local_oauth_server(
        verifier,
        callback_port=ANTHROPIC_CALLBACK_PORT,
        callback_path="/callback",
        success_message="Anthropic authentication completed. You can close this window.",
    )
    on_auth(
        OAuthAuthInfo(
            url=url,
            instructions=(
                "Complete login in your browser. If the browser is on another machine, "
                "paste the final redirect URL here."
            ),
        )
    )
    if open_browser:
        webbrowser.open(url)

    try:
        value = await _wait_for_input(server, on_manual_code_input)
        if value is None:
            value = await on_prompt(
                OAuthPrompt(
                    message="Paste the authorization code or full redirect URL:",
                    placeholder=ANTHROPIC_REDIRECT_URI,
                )
            )
        parsed = parse_authorization_input(value)
        if parsed.state is not None and parsed.state != verifier:
            raise OAuthError("OAuth state mismatch")
        if not parsed.code:
            raise OAuthError("Missing authorization code")
        on_progress and on_progress("Exchanging authorization code for tokens...")
        return await _anthropic_token_request(
            {
                "grant_type": "authorization_code",
                "client_id": ANTHROPIC_CLIENT_ID,
                "code": parsed.code,
                "state": parsed.state or verifier,
                "redirect_uri": ANTHROPIC_REDIRECT_URI,
                "code_verifier": verifier,
            },
            client=client,
            action="exchange",
        )
    finally:
        if server is not None:
            server.close()


async def refresh_anthropic_token(
    refresh_token: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> OAuthCredential:
    # [FUNCTION DOC]
    # Exchange a refresh token for new credentials via the token endpoint
    # (refresh grant). Keeps the old refresh token if the response omits one.
    # Input: refresh token, optional HTTP client. Output: OAuth credential.
    # Raises OAuthError on failure. Side effect: network request.
    """Refresh Anthropic OAuth credentials."""
    return await _anthropic_token_request(
        {
            "grant_type": "refresh_token",
            "client_id": ANTHROPIC_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        client=client,
        action="refresh",
        previous_refresh=refresh_token,
    )


async def _anthropic_token_request(
    data: dict[str, str],
    *,
    client: httpx.AsyncClient | None,
    action: str,
    previous_refresh: str | None = None,
) -> OAuthCredential:
    # [FUNCTION DOC]
    # HELPER: POST a JSON token request to Anthropic's token endpoint and parse
    # the credential. Inputs: request fields, optional client (a temporary one
    # with 30 s timeout is created and closed if absent), action label for error
    # messages, optional previous refresh token fallback. On HTTP status >= 400
    # raises OAuthError with a scrubbed error detail. Requires a JSON object
    # with a non-empty access token, a refresh token (or fallback), and a positive
    # numeric expires-in. Output: credential whose expiry is now + expires-in
    # minus a 5-minute skew, in epoch milliseconds.
    # Raises OAuthError on malformed responses; JSON decode errors propagate.
    owns_client = client is None
    active_client = client or create_async_client(timeout=30)
    try:
        response = await active_client.post(
            ANTHROPIC_TOKEN_URL,
            json=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
    finally:
        if owns_client:
            await active_client.aclose()
    if response.status_code >= 400:
        detail = _error_detail(
            response,
            secrets=[data[field] for field in _SECRET_REQUEST_FIELDS if field in data],
        )
        raise OAuthError(f"Anthropic token {action} failed ({response.status_code}): {detail}")
    raw = response.json()
    if not isinstance(raw, dict):
        raise OAuthError(f"Anthropic token {action} response must be an object")
    access = _required_string(raw, "access_token", action=action)
    refresh = _optional_string(raw, "refresh_token") or previous_refresh
    expires_in = raw.get("expires_in")
    if refresh is None:
        raise OAuthError(f"Anthropic token {action} response missing refresh_token")
    if not isinstance(expires_in, int | float) or isinstance(expires_in, bool) or expires_in <= 0:
        raise OAuthError(f"Anthropic token {action} response missing expires_in")
    return OAuthCredential(
        access=access,
        refresh=refresh,
        expires=int(time.time() * 1000 + expires_in * 1000 - ANTHROPIC_TOKEN_SKEW_MS),
    )


def _error_detail(response: httpx.Response, *, secrets: Iterable[str] = ()) -> str:
    # [FUNCTION DOC]
    # HELPER: summarize a failed token response safely. Parses the JSON body
    # (non-JSON -> no detail), extracts structured error fields only, replaces
    # any sent secret of length at least 8 with a redaction marker, and truncates
    # to 200 characters. Returns "no error detail" when nothing usable is found.
    """Summarize a token-endpoint failure body.

    The endpoint explains itself ("invalid_grant: Refresh token not found or
    invalid"); reporting only the status code turns a self-describing failure
    into a guessing game. Only the structured OAuth error fields are surfaced —
    the raw body stays out of the message because a failing request can echo
    the token it was sent. Those fields are server-controlled too, so anything
    we sent is scrubbed back out and the result is truncated before it reaches
    a log or the TUI.
    """
    try:
        raw = response.json()
    except ValueError:
        raw = None
    detail = _error_fields(raw) if isinstance(raw, dict) else None
    if detail is None:
        return "no error detail"
    for secret in secrets:
        if len(secret) >= 8:
            detail = detail.replace(secret, "<redacted>")
    return detail[:200]


def _error_fields(raw: dict[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: extract "code: description" from either the flat OAuth error shape
    # (error + error description) or the nested Anthropic API shape (error object
    # with type and message; a top-level message is also accepted). Returns the
    # combined string, either part alone, or None.
    """Pull the error code and message out of either error envelope.

    OAuth failures arrive as flat ``error``/``error_description``, while the
    Anthropic API's own shape nests them under ``error`` as ``type``/``message``.
    """
    error = raw.get("error")
    description = raw.get("error_description") or raw.get("message")
    if isinstance(error, dict):
        description = description or error.get("message")
        error = error.get("type")
    if isinstance(error, str) and isinstance(description, str):
        return f"{error}: {description}"
    if isinstance(error, str):
        return error
    if isinstance(description, str):
        return description
    return None


async def _wait_for_input(
    server: Any,
    manual_callback: Callable[[], Awaitable[str]] | None,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: race the local callback server's code wait against the manual
    # input callback. Creates a task for each available source; returns None if
    # none. Waits for the first to complete, cancels the rest, tells the server to
    # stop waiting, and returns the first finished task's result (which may be
    # None). Exceptions from the finished task propagate.
    tasks: list[asyncio.Task[str | None]] = []
    if server is not None:
        tasks.append(asyncio.create_task(server.wait_for_code()))
    if manual_callback is not None:
        tasks.append(asyncio.create_task(_manual_value(manual_callback)))
    if not tasks:
        return None
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if server is not None:
        server.cancel_wait()
    return next(iter(done)).result()


async def _manual_value(callback: Callable[[], Awaitable[str]]) -> str:
    # [FUNCTION DOC]
    # HELPER: await the manual input callback and return its string (wraps it as
    # a coroutine suitable for a task).
    return await callback()


def _required_string(raw: dict[str, Any], name: str, *, action: str) -> str:
    # [FUNCTION DOC]
    # HELPER: return the named non-empty string field from a response mapping,
    # else raise OAuthError mentioning the action and field.
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise OAuthError(f"Anthropic token {action} response missing {name}")
    return value


def _optional_string(raw: dict[str, Any], name: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: return the named field if it is a non-empty string, else None.
    value = raw.get(name)
    return value if isinstance(value, str) and value else None


class AnthropicOAuthProvider:
    # [CLASS DOC]
    # Registered OAuth provider object for Anthropic subscriptions: id
    # "anthropic", display name, browser flow kind. Stateless; delegates to the
    # module functions.
    """Registered Anthropic Claude subscription OAuth behavior."""

    id = ANTHROPIC_OAUTH_PROVIDER
    name = "Anthropic (Claude Pro/Max)"
    flow_kinds: tuple[OAuthFlowKind, ...] = ("browser",)

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredential:
        # [FUNCTION DOC]
        # Run the browser login using the given login callbacks (auth, prompt,
        # manual code input, progress). Returns the new credential; errors propagate.
        return await login_anthropic(
            on_auth=callbacks.on_auth,
            on_prompt=callbacks.on_prompt,
            on_manual_code_input=callbacks.on_manual_code_input,
            on_progress=callbacks.on_progress,
        )

    async def refresh(self, credential: OAuthCredential) -> OAuthCredential:
        # [FUNCTION DOC]
        # Return the credential unchanged if not expired; otherwise refresh it via
        # the token endpoint and return the new credential. May raise OAuthError.
        if not oauth_credential_is_expired(credential):
            return credential
        return await refresh_anthropic_token(credential.refresh)

    def runtime_auth(self, credential: OAuthCredential) -> OAuthRuntimeAuth:
        # [FUNCTION DOC]
        # Build runtime auth from a credential: the access token as API key plus
        # headers for bearer authorization, the Claude Code / OAuth beta flags, a
        # claude-cli user agent and an x-app cli header. No side effects.
        return OAuthRuntimeAuth(
            api_key=credential.access,
            headers={
                "Authorization": f"Bearer {credential.access}",
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                "user-agent": "claude-cli/tau",
                "x-app": "cli",
            },
        )
