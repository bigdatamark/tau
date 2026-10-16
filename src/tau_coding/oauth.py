# [MODULE DOC]
# OAuth support for the OpenAI Codex (ChatGPT subscription) provider: PKCE
# authorization-code flow with a local loopback callback server on port 1455
# (host overridable via the TAU_OAUTH_CALLBACK_HOST environment variable),
# optional manual paste of the code/redirect URL, token exchange and refresh
# against the OpenAI token endpoint, and extraction of the ChatGPT account id
# and expiry from the access token JWT. Also defines the provider object that
# plugs into the OAuth provider registry. Expiry times are epoch milliseconds;
# credentials are treated as expired 60 seconds early.
"""OAuth helpers for subscription-backed coding providers."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import threading
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from json import dumps, loads
from os import environ
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from tau_ai.http import create_async_client
from tau_coding.credentials import OAuthCredential
from tau_coding.oauth_types import (
    OAuthAuthInfo,
    OAuthFlowKind,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthRuntimeAuth,
)

OPENAI_CODEX_OAUTH_PROVIDER = "openai-codex"
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_ACCOUNT_CLAIM = "https://api.openai.com/auth"
OPENAI_CODEX_CALLBACK_PORT = 1455
TOKEN_REFRESH_SKEW_MS = 60_000

type AuthCallback = Callable[["OAuthAuthInfo"], None]
type PromptCallback = Callable[["OAuthPrompt"], Awaitable[str]]
type ManualCodeCallback = Callable[[], Awaitable[str]]
type ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AuthorizationCode:
    # [CLASS DOC]
    # Immutable parsed authorization callback data: optional code and optional
    # state.
    """Parsed OAuth authorization callback data."""

    code: str | None = None
    state: str | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationFlow:
    # [CLASS DOC]
    # Immutable state of one authorization attempt: PKCE verifier, random state
    # token and the full authorization URL.
    """OpenAI Codex OAuth authorization flow state."""

    verifier: str
    state: str
    url: str


@dataclass(frozen=True, slots=True)
class TokenResponse:
    # [CLASS DOC]
    # Immutable successful token-exchange result: access token, refresh token and
    # absolute expiry in epoch milliseconds.
    """Successful OAuth token response."""

    access: str
    refresh: str
    expires: int


class OAuthError(RuntimeError):
    # [CLASS DOC]
    # Exception (RuntimeError subclass) raised when an OAuth flow, token exchange
    # or refresh cannot complete.
    """Raised when an OAuth flow cannot complete."""


class OpenAICodexOAuthProvider:
    # [CLASS DOC]
    # OAuth provider implementation registered for "openai-codex" with display
    # name "OpenAI Codex (ChatGPT subscription)" and only the browser flow kind.
    # Stateless; delegates to the module functions.
    """Registered OpenAI Codex subscription OAuth behavior."""

    id = OPENAI_CODEX_OAUTH_PROVIDER
    name = "OpenAI Codex (ChatGPT subscription)"
    flow_kinds: tuple[OAuthFlowKind, ...] = ("browser",)

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredential:
        # [FUNCTION DOC]
        # Runs the browser login using the callbacks' on_auth, on_prompt, manual-code
        # and progress callbacks. Output: the new credential.
        # Raises OAuthError when the device_code method is requested (unsupported) and
        # propagates any login error. Side effects: opens a browser, binds a local
        # port, performs network requests.
        if callbacks.method == "device_code":
            raise OAuthError("OpenAI Codex device-code login is not implemented in Tau yet")
        return await login_openai_codex(
            on_auth=callbacks.on_auth,
            on_prompt=callbacks.on_prompt,
            on_manual_code_input=callbacks.on_manual_code_input,
            on_progress=callbacks.on_progress,
        )

    async def refresh(self, credential: OAuthCredential) -> OAuthCredential:
        # [FUNCTION DOC]
        # Returns the credential unchanged when not yet expired (with skew), otherwise
        # refreshes it via the token endpoint and returns the new credential.
        # Raises OAuthError on refresh failure. Side effect: network request.
        if not oauth_credential_is_expired(credential):
            return credential
        return await refresh_openai_codex_token(credential.refresh)

    def runtime_auth(self, credential: OAuthCredential) -> OAuthRuntimeAuth:
        # [FUNCTION DOC]
        # Returns runtime auth whose API key is the credential's access token.
        return OAuthRuntimeAuth(api_key=credential.access)


class _LocalOAuthServer:
    # [CLASS DOC]
    # HELPER: handle wrapping a running threaded HTTP callback server, its serving
    # thread and an asyncio future resolved with the received code (or none).
    def __init__(
        self,
        server: ThreadingHTTPServer,
        thread: threading.Thread,
        future: asyncio.Future[str | None],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the server, thread and future. No side effects.
        self._server = server
        self._thread = thread
        self._future = future

    async def wait_for_code(self) -> str | None:
        # [FUNCTION DOC]
        # HELPER: awaits the future; returns the code, or none if the wait was
        # cancelled via cancel_wait.
        """Wait for the local callback server to receive a code."""
        return await self._future

    def cancel_wait(self) -> None:
        # [FUNCTION DOC]
        # HELPER: resolves the future with none if it is still pending.
        """Resolve the pending wait without an authorization code."""
        if not self._future.done():
            self._future.set_result(None)

    def close(self) -> None:
        # [FUNCTION DOC]
        # HELPER: shuts down the server loop, closes the socket and joins the thread
        # for up to one second. Blocking call.
        """Stop the local callback server."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)


def create_pkce_pair() -> tuple[str, str]:
    # [FUNCTION DOC]
    # Generates a random URL-safe verifier (64 random bytes encoded) and its S256
    # challenge (SHA-256 digest, unpadded base64url). Output: verifier and
    # challenge pair.
    """Return a PKCE verifier and S256 challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64url(digest)
    return verifier, challenge


def create_openai_codex_authorization_flow(
    *,
    originator: str = "tau",
) -> AuthorizationFlow:
    # [FUNCTION DOC]
    # Creates a PKCE pair and a 32-hex-character random state and builds the
    # authorize URL with response type code, client id, redirect URI, scopes,
    # S256 challenge, state, organization/simplified-flow flags and the given
    # originator (default "tau"). Output: the authorization flow.
    """Create an OpenAI Codex OAuth authorization URL."""
    verifier, challenge = create_pkce_pair()
    state = secrets.token_hex(16)
    params = {
        "response_type": "code",
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
        "scope": OPENAI_CODEX_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    return AuthorizationFlow(
        verifier=verifier,
        state=state,
        url=f"{OPENAI_CODEX_AUTHORIZE_URL}?{urlencode(params)}",
    )


def parse_authorization_input(value: str) -> AuthorizationCode:
    # [FUNCTION DOC]
    # Parses user input into a code/state pair. Blank input gives both none. A
    # full URL with scheme and host yields its code/state query values. Text with
    # "#" is split into code and state at the first "#". Text containing "code="
    # is parsed as a query string. Anything else is treated as a raw code.
    # Empty parts become none. No side effects.
    """Parse a pasted redirect URL, query string, code#state pair, or raw code."""
    stripped = value.strip()
    if not stripped:
        return AuthorizationCode()

    parsed_url = urlparse(stripped)
    if parsed_url.scheme and parsed_url.netloc:
        params = parse_qs(parsed_url.query)
        return AuthorizationCode(
            code=_first_query_value(params, "code"),
            state=_first_query_value(params, "state"),
        )

    if "#" in stripped:
        code, state = stripped.split("#", 1)
        return AuthorizationCode(code=code or None, state=state or None)

    if "code=" in stripped:
        params = parse_qs(stripped)
        return AuthorizationCode(
            code=_first_query_value(params, "code"),
            state=_first_query_value(params, "state"),
        )

    return AuthorizationCode(code=stripped)


def oauth_credential_is_expired(credential: OAuthCredential) -> bool:
    # [FUNCTION DOC]
    # Returns true when current time in ms is at or past the credential expiry
    # minus the 60-second skew.
    """Return whether an OAuth credential should be refreshed before use."""
    return int(time.time() * 1000) >= credential.expires - TOKEN_REFRESH_SKEW_MS


async def login_openai_codex(
    *,
    on_auth: AuthCallback,
    on_prompt: PromptCallback,
    on_manual_code_input: ManualCodeCallback | None = None,
    on_progress: ProgressCallback | None = None,
    open_browser: bool = True,
    originator: str = "tau",
    client: httpx.AsyncClient | None = None,
) -> OAuthCredential:
    # [FUNCTION DOC]
    # Runs the full Codex login.
    # Inputs (keyword): on_auth (receives URL and instructions), on_prompt (asks
    # for pasted input), optional manual-code callback and progress callback,
    # open_browser flag (default true), originator and optional HTTP client.
    # Behavior: creates a flow, starts the local callback server (may be absent if
    # the port is busy), reports the URL, optionally opens the browser, then waits
    # for either the server callback or the manual-code callback. If neither
    # yields a code, prompts the user to paste the code or redirect URL, validating
    # state. Reports progress, exchanges the code, and extracts the account id.
    # Output: credential with access, refresh, expiry and account id.
    # Raises OAuthError for missing code, state mismatch, failed exchange or
    # missing account id. Side effects: binds a port, starts a thread, opens a
    # browser, network requests; always closes the server.
    # KNOWN ISSUE: state validation accepts input with no state at all, so a
    # pasted bare code bypasses the CSRF state check.
    """Run OpenAI Codex OAuth and return refreshable credentials."""
    flow = create_openai_codex_authorization_flow(originator=originator)
    server = await _start_local_oauth_server(flow.state)

    on_auth(
        OAuthAuthInfo(
            url=flow.url,
            instructions="A browser window should open. Complete login to finish.",
        )
    )
    if open_browser:
        webbrowser.open(flow.url)

    try:
        code = await _wait_for_authorization_code(
            flow=flow,
            server=server,
            on_manual_code_input=on_manual_code_input,
        )
        if code is None:
            manual_input = await on_prompt(
                OAuthPrompt(message="Paste the authorization code or full redirect URL:")
            )
            parsed = parse_authorization_input(manual_input)
            _validate_state(parsed.state, flow.state)
            code = parsed.code
        if not code:
            raise OAuthError("Missing authorization code")
        on_progress and on_progress("Exchanging authorization code...")
        token = await exchange_openai_codex_authorization_code(
            code,
            flow.verifier,
            client=client,
        )
        account_id = account_id_from_access_token(token.access)
        if account_id is None:
            raise OAuthError("Failed to extract OpenAI account id from access token")
        return OAuthCredential(
            access=token.access,
            refresh=token.refresh,
            expires=token.expires,
            account_id=account_id,
        )
    finally:
        if server is not None:
            server.close()


async def exchange_openai_codex_authorization_code(
    code: str,
    verifier: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> TokenResponse:
    # [FUNCTION DOC]
    # Posts an authorization_code grant (client id, code, PKCE verifier, redirect
    # URI) to the token endpoint. Output: token response with access, refresh and
    # expiry. Raises OAuthError on HTTP error, non-object JSON, missing tokens or
    # missing/invalid expiry. Side effect: network request.
    """Exchange an OpenAI Codex authorization code for OAuth tokens."""
    raw = await _post_openai_codex_token(
        {
            "grant_type": "authorization_code",
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
        },
        client=client,
        action="exchange",
    )
    access_token = _required_token_field(raw, "access_token", action="exchange")
    refresh_token = _required_token_field(raw, "refresh_token", action="exchange")
    return TokenResponse(
        access=access_token,
        refresh=refresh_token,
        expires=_token_expiry(raw, access_token, action="exchange"),
    )


async def refresh_openai_codex_token(
    refresh_token: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> OAuthCredential:
    # [FUNCTION DOC]
    # Posts a refresh_token grant. Output: a new credential whose refresh token is
    # the returned one or, if absent, the old one; expiry from expires_in or the
    # JWT; account id from the new access token. Raises OAuthError on failure or
    # if no account id can be extracted. Side effect: network request.
    """Refresh OpenAI Codex OAuth credentials."""
    raw = await _post_openai_codex_token(
        {
            "grant_type": "refresh_token",
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        client=client,
        action="refresh",
    )
    access_token = _required_token_field(raw, "access_token", action="refresh")
    next_refresh_token = _optional_token_field(raw, "refresh_token") or refresh_token
    account_id = account_id_from_access_token(access_token)
    if account_id is None:
        raise OAuthError("Failed to extract OpenAI account id from refreshed access token")
    return OAuthCredential(
        access=access_token,
        refresh=next_refresh_token,
        expires=_token_expiry(raw, access_token, action="refresh"),
        account_id=account_id,
    )


def account_id_from_access_token(access_token: str) -> str | None:
    # [FUNCTION DOC]
    # Decodes the JWT payload (without signature verification) and returns the
    # stripped chatgpt_account_id inside the OpenAI auth claim, or none if the
    # token, claim or id is missing or invalid.
    """Extract the ChatGPT account id from an OpenAI Codex access JWT."""
    payload = _access_token_payload(access_token)
    if payload is None:
        return None
    auth = payload.get(OPENAI_CODEX_ACCOUNT_CLAIM)
    if not isinstance(auth, dict):
        return None
    account_id = auth.get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id.strip():
        return None
    return account_id.strip()


def _access_token_expiry(access_token: str) -> int | None:
    # [FUNCTION DOC]
    # HELPER: returns the JWT "exp" claim (positive number, not boolean) converted
    # to epoch milliseconds, or none.
    payload = _access_token_payload(access_token)
    if payload is None:
        return None
    exp = payload.get("exp")
    if isinstance(exp, int | float) and not isinstance(exp, bool) and exp > 0:
        return int(exp * 1000)
    return None


def _access_token_payload(access_token: str) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: splits a three-part JWT, base64url-decodes and JSON-parses the
    # middle part, returning it if it is a mapping; any failure returns none.
    # No signature verification.
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            return None
        payload = loads(_base64url_decode(parts[1]).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


async def _post_openai_codex_token(
    data: dict[str, str],
    *,
    client: httpx.AsyncClient | None,
    action: str,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: posts form-encoded data to the token endpoint using the given client
    # or a temporary one (60 s timeout, closed afterwards).
    # Output: the decoded JSON object. Raises OAuthError for status 400 or above
    # (including the response body) or when the JSON is not an object.
    # KNOWN ISSUE: a non-JSON body raises a raw decode error rather than
    # OAuthError, and network errors propagate as httpx exceptions.
    owns_client = client is None
    active_client = client or create_async_client(timeout=60)
    try:
        response = await active_client.post(
            OPENAI_CODEX_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    finally:
        if owns_client:
            await active_client.aclose()

    if response.status_code >= 400:
        raise OAuthError(
            f"OpenAI Codex token {action} failed ({response.status_code}): {response.text}"
        )

    raw = response.json()
    if not isinstance(raw, dict):
        raise OAuthError(f"OpenAI Codex token {action} response must be a JSON object")
    return raw


def _required_token_field(raw: dict[str, Any], field: str, *, action: str) -> str:
    # [FUNCTION DOC]
    # HELPER: returns a non-empty string field from the token response or raises
    # OAuthError naming the action and field and dumping the whole response
    # (which may include secrets in error messages).
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise OAuthError(
            f"OpenAI Codex token {action} response missing {field}: {dumps(raw, sort_keys=True)}"
        )
    return value


def _optional_token_field(raw: dict[str, Any], field: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: returns a non-empty string field or none.
    value = raw.get(field)
    if isinstance(value, str) and value:
        return value
    return None


def _token_expiry(raw: dict[str, Any], access_token: str, *, action: str) -> int:
    # [FUNCTION DOC]
    # HELPER: computes absolute expiry in ms: now plus numeric expires_in seconds;
    # raises OAuthError when expires_in is present but not numeric; otherwise uses
    # the JWT exp claim; raises OAuthError if neither is available.
    expires_in = raw.get("expires_in")
    if isinstance(expires_in, int | float) and not isinstance(expires_in, bool):
        return int(time.time() * 1000) + int(expires_in * 1000)
    if expires_in is not None:
        raise OAuthError(
            f"OpenAI Codex token {action} response has invalid expires_in: "
            f"{dumps(raw, sort_keys=True)}"
        )
    expires = _access_token_expiry(access_token)
    if expires is not None:
        return expires
    raise OAuthError(
        f"OpenAI Codex token {action} response missing expiry: {dumps(raw, sort_keys=True)}"
    )


async def _wait_for_authorization_code(
    *,
    flow: AuthorizationFlow,
    server: _LocalOAuthServer | None,
    on_manual_code_input: ManualCodeCallback | None,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: races the local server's code wait and the manual-code callback
    # (whichever exist) and takes the first finished result, cancelling the other
    # and always resolving the server wait. Returns none when no source exists or
    # the winner produced none; otherwise parses the input, validates state
    # (raising OAuthError on mismatch) and returns the code (may be none).
    # Exceptions from the winning task propagate. Cancelled tasks are not awaited.
    tasks: list[asyncio.Task[str | None]] = []
    if server is not None:
        tasks.append(asyncio.create_task(server.wait_for_code()))
    if on_manual_code_input is not None:
        tasks.append(asyncio.create_task(_await_manual_code(on_manual_code_input)))
    if not tasks:
        return None

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        result = next(iter(done)).result()
    finally:
        if server is not None:
            server.cancel_wait()

    if result is None:
        return None
    parsed = parse_authorization_input(result)
    _validate_state(parsed.state, flow.state)
    return parsed.code


async def _await_manual_code(callback: ManualCodeCallback) -> str | None:
    # [FUNCTION DOC]
    # HELPER: awaits the manual-code callback and returns its string.
    return await callback()


def _validate_state(state: str | None, expected_state: str) -> None:
    # [FUNCTION DOC]
    # HELPER: raises OAuthError("OAuth state mismatch") when a state is present
    # and differs from the expected one; a missing state passes.
    if state is not None and state != expected_state:
        raise OAuthError("OAuth state mismatch")


async def _start_local_oauth_server(
    state: str,
    *,
    callback_port: int = OPENAI_CODEX_CALLBACK_PORT,
    callback_path: str = "/auth/callback",
    success_message: str = "OpenAI authentication completed. You can close this window.",
) -> _LocalOAuthServer | None:
    # [FUNCTION DOC]
    # HELPER: starts a threaded HTTP server on the configured host and callback
    # port that handles the OAuth redirect.
    # Inputs: expected state and keyword port, callback path and success message.
    # Output: a server handle, or none if binding fails with OSError.
    # Side effects: binds a socket and starts a daemon serving thread; the
    # request handler resolves the future on the event loop thread-safely.
    host = environ.get("TAU_OAUTH_CALLBACK_HOST", "127.0.0.1")
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str | None] = loop.create_future()

    class CallbackHandler(BaseHTTPRequestHandler):
        # [CLASS DOC]
        # HELPER: request handler class closed over the expected state, path, loop and
        # future.
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            # [FUNCTION DOC]
            # HELPER: handles a GET: wrong path gives 404, state mismatch 400, missing
            # code 400; otherwise responds 200 with the success page and resolves the
            # future with the code (first one wins). Unexpected errors produce 500. All
            # responses are small HTML pages.
            try:
                parsed = urlparse(self.path)
                if parsed.path != callback_path:
                    self._finish(404, _oauth_html("Callback route not found."))
                    return
                params = parse_qs(parsed.query)
                if _first_query_value(params, "state") != state:
                    self._finish(400, _oauth_html("OAuth state mismatch."))
                    return
                code = _first_query_value(params, "code")
                if not code:
                    self._finish(400, _oauth_html("Missing authorization code."))
                    return
                self._finish(
                    200,
                    _oauth_html(success_message),
                )
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, code)
            except Exception:
                self._finish(500, _oauth_html("Internal error while processing OAuth callback."))

        def log_message(self, _format: str, *_args: Any) -> None:
            # [FUNCTION DOC]
            # HELPER: suppresses the default request logging to stderr.
            return

        def _finish(self, status: int, body: str) -> None:
            # [FUNCTION DOC]
            # HELPER: writes a status line, HTML content-type and length headers and the
            # UTF-8 body.
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    try:
        server = ThreadingHTTPServer((host, callback_port), CallbackHandler)
    except OSError:
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return _LocalOAuthServer(server, thread, future)


def _first_query_value(params: dict[str, list[str]], key: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: returns the first value for a key in parsed query parameters, or
    # none when absent or empty.
    values = params.get(key)
    if not values:
        return None
    return values[0] or None


def _base64url(value: bytes) -> str:
    # [FUNCTION DOC]
    # HELPER: unpadded URL-safe base64 encoding of bytes to an ASCII string.
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64url_decode(value: str) -> bytes:
    # [FUNCTION DOC]
    # HELPER: URL-safe base64 decoding that re-adds missing padding. Raises on
    # invalid input.
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _oauth_html(message: str) -> str:
    # [FUNCTION DOC]
    # HELPER: returns a minimal HTML page titled "Tau OAuth" containing the
    # HTML-escaped message in a paragraph.
    escaped = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<!doctype html><meta charset="utf-8"><title>Tau OAuth</title><p>{escaped}</p>'
