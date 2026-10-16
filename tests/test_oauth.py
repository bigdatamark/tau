# [MODULE DOC]
# Tests for the OpenAI Codex OAuth helpers in tau_coding.oauth: authorization
# URL construction with PKCE, parsing of pasted authorization input, account
# id extraction from JWT access tokens, and token refresh behavior using an
# httpx mock transport (no real network).
import base64
from json import dumps
from time import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from tau_coding.oauth import (
    OPENAI_CODEX_ACCOUNT_CLAIM,
    OPENAI_CODEX_CLIENT_ID,
    account_id_from_access_token,
    create_openai_codex_authorization_flow,
    parse_authorization_input,
    refresh_openai_codex_token,
)


def test_create_openai_codex_authorization_flow_includes_pkce_and_codex_params() -> None:
    # [FUNCTION DOC]
    # Verify the generated authorize URL targets auth.openai.com and includes the
    # code response type, client id, localhost:1455 redirect, expected scopes,
    # S256 challenge method, Codex simplified-flow flag, originator, state
    # matching the flow, a non-empty challenge, and a verifier.
    flow = create_openai_codex_authorization_flow(originator="tau-test")

    url = urlparse(flow.url)
    params = parse_qs(url.query)

    assert url.geturl().startswith("https://auth.openai.com/oauth/authorize?")
    assert params["response_type"] == ["code"]
    assert params["client_id"] == [OPENAI_CODEX_CLIENT_ID]
    assert params["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert params["scope"] == ["openid profile email offline_access"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["codex_cli_simplified_flow"] == ["true"]
    assert params["originator"] == ["tau-test"]
    assert params["state"] == [flow.state]
    assert params["code_challenge"][0]
    assert flow.verifier


def test_parse_authorization_input_accepts_redirect_url_query_and_raw_code() -> None:
    # [FUNCTION DOC]
    # Verify authorization input parsing accepts a full redirect URL, a bare
    # query string, a "code#state" form, and a raw code.
    assert (
        parse_authorization_input("http://localhost:1455/auth/callback?code=abc&state=state-1").code
        == "abc"
    )
    assert parse_authorization_input("code=abc&state=state-1").state == "state-1"
    assert parse_authorization_input("abc#state-1").state == "state-1"
    assert parse_authorization_input("abc").code == "abc"


def test_account_id_from_access_token_reads_openai_auth_claim() -> None:
    # [FUNCTION DOC]
    # Verify the account id is read from the OpenAI auth claim of a JWT and that
    # a non-JWT string yields None.
    assert account_id_from_access_token(_jwt("account-1")) == "account-1"
    assert account_id_from_access_token("not-a-jwt") is None


@pytest.mark.anyio
async def test_refresh_openai_codex_token_returns_oauth_credential() -> None:
    # [FUNCTION DOC]
    # Verify refresh posts a form body with refresh grant and client id, and the
    # returned credential carries the new access/refresh tokens, the account id
    # from the JWT, and a positive expiry.
    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: mock transport handler asserting the request body and returning a
        # 200 JSON token response with a new JWT, new refresh token, and expires-in.
        body = request.content.decode()
        assert "grant_type=refresh_token" in body
        assert "client_id=" in body
        return httpx.Response(
            200,
            json={
                "access_token": _jwt("account-2"),
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        credential = await refresh_openai_codex_token("old-refresh", client=client)

    assert credential.access == _jwt("account-2")
    assert credential.refresh == "new-refresh"
    assert credential.account_id == "account-2"
    assert credential.expires > 0


@pytest.mark.anyio
async def test_refresh_openai_codex_token_preserves_refresh_and_reads_jwt_expiry() -> None:
    # [FUNCTION DOC]
    # Verify that when the response omits refresh token and expires-in, the old
    # refresh token is kept and expiry comes from the JWT exp claim (in ms).
    expires = int(time()) + 3600
    access_token = _jwt("account-3", expires=expires)

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: mock handler asserting a refresh grant and returning only an access
        # token.
        body = request.content.decode()
        assert "grant_type=refresh_token" in body
        return httpx.Response(200, json={"access_token": access_token})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        credential = await refresh_openai_codex_token("old-refresh", client=client)

    assert credential.access == access_token
    assert credential.refresh == "old-refresh"
    assert credential.account_id == "account-3"
    assert credential.expires == expires * 1000


def _jwt(account_id: str, *, expires: int | None = None) -> str:
    # [FUNCTION DOC]
    # HELPER: build an unsigned JWT-like string (none-alg header, payload with
    # the OpenAI auth claim containing the account id and optional exp, literal
    # "signature") using unpadded base64url segments.
    payload = {OPENAI_CODEX_ACCOUNT_CLAIM: {"chatgpt_account_id": account_id}}
    if expires is not None:
        payload["exp"] = expires
    return ".".join(
        [
            _base64url(dumps({"alg": "none"}).encode()),
            _base64url(dumps(payload).encode()),
            "signature",
        ]
    )


def _base64url(value: bytes) -> str:
    # [FUNCTION DOC]
    # HELPER: base64url-encode bytes and strip padding, returning ASCII text.
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
