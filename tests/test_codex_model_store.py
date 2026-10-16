# [MODULE DOC]
# Tests for the Codex model-catalog cache store in tau_coding.codex_model_store.
# Covers round-tripping a runtime model catalog to disk scoped by account id, and
# rejection (returning None) of malformed cached documents.
import json
from pathlib import Path

import pytest

from tau_ai.model_catalog import RuntimeModel, RuntimeModelCatalog
from tau_ai.model_limits import RuntimeModelLimits
from tau_coding.codex_model_store import (
    cached_codex_model_catalog,
    save_codex_model_catalog,
)
from tau_coding.paths import TauPaths


def test_codex_model_catalog_round_trips_with_account_scope(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Saves a one-model catalog (with limits, image modality and thinking levels) for
    # account "account-1" at a fixed timestamp, then asserts: the returned path is the
    # configured Codex model store path; loading for a different account returns None;
    # loading for the same account returns an equal catalog. Writes files under tmp_path.
    paths = TauPaths(home=tmp_path / ".tau")
    catalog = RuntimeModelCatalog(
        (
            RuntimeModel(
                id="gpt-live",
                name="GPT Live",
                limits=RuntimeModelLimits(
                    context_window=400_000,
                    max_output_tokens=100_000,
                    effective_context_window_percent=95,
                    auto_compact_token_limit=350_000,
                ),
                input_modalities=("text", "image"),
                thinking_levels=("low", "high"),
                default_thinking_level="high",
            ),
        )
    )

    path = save_codex_model_catalog(catalog, account_id="account-1", paths=paths, now=123.0)

    assert path == paths.codex_models_store_path
    assert cached_codex_model_catalog(paths, account_id="other-account") is None
    assert cached_codex_model_catalog(paths, account_id="account-1") == catalog


@pytest.mark.parametrize("field", ["input_modalities", "thinking_levels"])
@pytest.mark.parametrize("invalid_item", [{}, [], None, 1, True, "unknown"])
def test_codex_model_catalog_cache_rejects_invalid_model_values(
    tmp_path: Path, field: str, invalid_item: object
) -> None:
    # [FUNCTION DOC]
    # Parametrised over the list fields input_modalities and thinking_levels and over
    # invalid items (empty mapping, empty list, None, int, bool, unknown string).
    # Writes a schema-version-1 cache document whose model has one invalid item in the
    # chosen field and asserts the loader returns None rather than raising.
    paths = TauPaths(home=tmp_path)
    model = {
        "id": "gpt-live",
        "input_modalities": ["text"],
        "thinking_levels": ["low"],
    }
    document = {
        "schema_version": 1,
        "account_id": "account-1",
        "catalog": {"models": [model]},
    }
    model[field] = [invalid_item]
    paths.codex_models_store_path.write_text(json.dumps(document), encoding="utf-8")

    assert cached_codex_model_catalog(paths, account_id="account-1") is None


def test_codex_model_catalog_cache_rejects_invalid_documents(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Writes a well-formed schema-version-1 document whose catalog has an empty model list
    # and asserts the loader treats it as invalid (returns None), i.e. empty catalogs are
    # not accepted from cache.
    paths = TauPaths(home=tmp_path / ".tau")
    paths.codex_models_store_path.parent.mkdir(parents=True)
    paths.codex_models_store_path.write_text(
        '{"schema_version": 1, "account_id": "account-1", "catalog": {"models": []}}',
        encoding="utf-8",
    )

    assert cached_codex_model_catalog(paths, account_id="account-1") is None
