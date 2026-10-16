# [MODULE DOC]
# Tests for the file-based credential store: API key round trips with private file permissions,
# deletion, empty-value rejection, OAuth credential storage (including extensible metadata and
# legacy Codex shape), and TAU_HOME isolation of the default store location.
from stat import S_IMODE

import pytest

from tau_coding.credentials import CredentialStoreError, FileCredentialStore, OAuthCredential


def test_file_credential_store_round_trips_and_sets_private_permissions(tmp_path) -> None:
    # [FUNCTION DOC]
    # Store an API key in a temp credentials file; verify it reads back and the file mode is owner
    # read/write only (0600).
    path = tmp_path / "credentials.json"
    store = FileCredentialStore(path)

    store.set("openai", "test-key")

    assert store.get("openai") == "test-key"
    assert S_IMODE(path.stat().st_mode) == 0o600


def test_file_credential_store_deletes_key(tmp_path) -> None:
    # [FUNCTION DOC]
    # Store then delete an API key; verify lookup returns None.
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set("openai", "test-key")

    store.delete("openai")

    assert store.get("openai") is None


def test_file_credential_store_rejects_empty_values(tmp_path) -> None:
    # [FUNCTION DOC]
    # Verify storing an empty API key raises the credential store error mentioning "must not be
    # empty".
    store = FileCredentialStore(tmp_path / "credentials.json")

    with pytest.raises(CredentialStoreError, match="must not be empty"):
        store.set("openai", "")


def test_file_credential_store_round_trips_oauth_credentials(tmp_path) -> None:
    # [FUNCTION DOC]
    # Store an OAuth credential with account id; verify it is not visible as an API key, reads
    # back equal via the OAuth getter, and is persisted with a type "oauth" marker in the JSON
    # file.
    path = tmp_path / "credentials.json"
    store = FileCredentialStore(path)
    credential = OAuthCredential(
        access="access-token",
        refresh="refresh-token",
        expires=123456,
        account_id="account-1",
    )

    store.set_oauth("openai-codex", credential)

    assert store.get("openai-codex") is None
    assert store.get_oauth("openai-codex") == credential
    assert '"type": "oauth"' in path.read_text(encoding="utf-8")


def test_tau_home_isolates_default_oauth_credential_stores(monkeypatch, tmp_path) -> None:
    # [FUNCTION DOC]
    # Using monkeypatch to switch TAU_HOME between two temp directories, store different OAuth
    # credentials with the default-path store and verify each home returns its own credential and
    # the default home directory file is never created.
    personal_home = tmp_path / ".tau-personal"
    work_home = tmp_path / ".tau-work"
    personal = OAuthCredential(access="personal", refresh="personal-refresh", expires=123456)
    work = OAuthCredential(access="work", refresh="work-refresh", expires=123456)

    monkeypatch.setenv("TAU_HOME", str(personal_home))
    FileCredentialStore().set_oauth("anthropic", personal)
    monkeypatch.setenv("TAU_HOME", str(work_home))
    FileCredentialStore().set_oauth("anthropic", work)

    assert FileCredentialStore().get_oauth("anthropic") == work
    monkeypatch.setenv("TAU_HOME", str(personal_home))
    assert FileCredentialStore().get_oauth("anthropic") == personal
    assert not (tmp_path / ".tau" / "credentials.json").exists()


def test_file_credential_store_round_trips_extensible_oauth_metadata(tmp_path) -> None:
    # [FUNCTION DOC]
    # Store an OAuth credential with arbitrary metadata (enterprise domain, model id list); verify
    # round trip equality, that no account id key is written when unset, and that no temporary
    # write files are left behind (atomic write cleanup).
    path = tmp_path / "credentials.json"
    store = FileCredentialStore(path)
    credential = OAuthCredential(
        access="copilot-access",
        refresh="github-token",
        expires=123456,
        metadata={
            "enterprise_domain": "ghe.example.com",
            "available_model_ids": ["gpt-5.4", "claude-sonnet-4.6"],
        },
    )

    store.set_oauth("github-copilot", credential)

    assert store.get_oauth("github-copilot") == credential
    assert '"account_id"' not in path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".credentials.json.*"))


def test_file_credential_store_loads_legacy_codex_oauth_shape(tmp_path) -> None:
    # [FUNCTION DOC]
    # Write a legacy JSON layout for a Codex OAuth credential with a top-level account id; verify
    # it loads as the equivalent OAuth credential.
    path = tmp_path / "credentials.json"
    path.write_text(
        '{"openai-codex":{"type":"oauth","access":"a","refresh":"r",'
        '"expires":123,"account_id":"account"}}',
        encoding="utf-8",
    )

    credential = FileCredentialStore(path).get_oauth("openai-codex")

    assert credential == OAuthCredential(
        access="a",
        refresh="r",
        expires=123,
        account_id="account",
    )
