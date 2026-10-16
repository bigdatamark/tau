# [MODULE DOC]
# Build-time generation and runtime loading of the models.dev-derived model
# catalog for Tau's bundled providers (Pi uses models.dev the same way).
# Constants: the models.dev API URL, NVIDIA's live model-list URL, the package
# resource path of the generated catalog JSON, a set of NVIDIA model ids known
# not to work, and a private mapping from Tau provider names to models.dev keys
# (kimi-code -> kimi-for-coding, together -> togetherai).
# Functions filter NVIDIA models, map reasoning effort options to thinking
# levels, build the catalog document, load the bundled document and convert it
# into a raw overlay for the provider catalog.
# Side effects: only the bundled loaders read a package resource file.
"""Build-time models.dev catalog generation for Tau's bundled providers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from importlib.resources import files
from typing import Any

from tau_coding.provider_catalog import ProviderCatalogEntry
from tau_coding.thinking import THINKING_LEVELS, ThinkingLevel

MODELS_DEV_URL = "https://models.dev/api.json"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
MODELS_DEV_CATALOG_RESOURCE = "data/models-dev-catalog.json"
NVIDIA_UNSUPPORTED_MODELS = {
    "abacusai/dracarys-llama-3.1-70b-instruct",
    "bytedance/seed-oss-36b-instruct",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "google/gemma-2-2b-it",
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it",
    "google/gemma-4-31b-it",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "microsoft/phi-4-mini-instruct",
    "minimaxai/minimax-m2.7",
    "mistralai/mistral-nemotron",
    "nvidia/nemotron-mini-4b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3.5-397b-a17b",
    "sarvamai/sarvam-m",
    "upstage/solar-10.7b-instruct",
}

# Tau names providers for users; models.dev names them for its own catalog.
_MODELS_DEV_PROVIDER_KEYS = {
    "kimi-code": "kimi-for-coding",
    "together": "togetherai",
}


def nvidia_model_filter(source: object, live_source: object) -> set[str]:
    # [FUNCTION DOC]
    # Computes which models.dev NVIDIA model ids are actually served by the live
    # NVIDIA NIM model list (Pi-compatible filter).
    # Inputs: the parsed models.dev document and the parsed live NVIDIA listing.
    # Behavior: normalises ids by lowercasing and replacing underscores with dots;
    # keeps source ids whose normalised form appears in the live ids and which are
    # not in the unsupported set.
    # Outputs: a set of original models.dev model ids.
    # Raises ValueError if either input is not a JSON object, or if the nvidia
    # models entry is not an object or the live "data" field is not a list.
    """Return models.dev NVIDIA IDs accepted by Pi's live NIM filter."""
    if not isinstance(source, dict) or not isinstance(live_source, dict):
        raise ValueError("NVIDIA filtering sources must be JSON objects")
    provider = source.get("nvidia")
    source_models = provider.get("models") if isinstance(provider, dict) else None
    live_models = live_source.get("data")
    if not isinstance(source_models, dict) or not isinstance(live_models, list):
        raise ValueError("NVIDIA filtering sources have invalid model data")
    live_ids = {
        model["id"].lower().replace("_", ".")
        for model in live_models
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    }
    return {
        model_id
        for model_id in source_models
        if model_id.lower().replace("_", ".") in live_ids
        and model_id not in NVIDIA_UNSUPPORTED_MODELS
    }


def thinking_level_map_from_reasoning_options(
    reasoning_options: object,
) -> dict[ThinkingLevel, str | None] | None:
    # [FUNCTION DOC]
    # Converts a models.dev "reasoning_options" list into a map from each Tau
    # thinking level to the provider effort value (or none when unsupported).
    # Inputs: the raw reasoning options value.
    # Behavior: collects all string/null values of options whose type is
    # "effort". Returns none if the input is not a list, no values were found, or
    # none of the selectable levels (all levels except off) nor "none" appear.
    # Otherwise maps "off" to "none" if available (else none) and every other
    # level to itself if listed (else none).
    # Outputs: the mapping or none. Pure; never raises.
    """Convert verified models.dev effort values using Pi's exact semantics."""
    if not isinstance(reasoning_options, list):
        return None

    values: set[str | None] = set()
    for option in reasoning_options:
        if not isinstance(option, Mapping) or option.get("type") != "effort":
            continue
        raw_values = option.get("values")
        if isinstance(raw_values, list):
            values.update(value for value in raw_values if isinstance(value, str) or value is None)

    # Pi emits no map when models.dev has no usable effort values. Toggle-only
    # and empty option arrays therefore retain provider/manual behavior.
    selectable = set(THINKING_LEVELS) - {"off"}
    if not values or not (selectable.intersection(values) or "none" in values):
        return None

    return {
        "off": "none" if "none" in values else None,
        **{
            level: level if level in values else None for level in THINKING_LEVELS if level != "off"
        },
    }


def models_dev_catalog_document(
    source: object,
    catalog: Iterable[ProviderCatalogEntry],
    *,
    provider_model_filters: Mapping[str, set[str]] | None = None,
    generated_at: int = 0,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Builds the full generated catalog document from models.dev data for the
    # given provider catalog entries.
    # Inputs: models.dev document; iterable of ProviderCatalogEntry; optional
    # mapping of provider name to allowed model id set; generation timestamp.
    # Behavior per provider (sorted by name): look up its models.dev entry (via the
    # key alias map); skip if absent/malformed. Eligible models = string ids with
    # object rows, allowed by the filter (if any) and passing the eligibility
    # check (tool calling, not deprecated, text in/out). Model order: existing
    # catalog models that are eligible, then catalog-only models (not in source)
    # and the provider default, then new eligible ids sorted; de-duplicated. Skip
    # provider if empty. Metadata is built for every eligible model, sorted by id.
    # Outputs: a document with schema_version 1, source URL, generated_at and a
    # providers mapping of name to models list and model_metadata.
    # Raises ValueError if the source is not a mapping. Pure otherwise.
    # KNOWN ISSUE: the docstring says ineligible source rows are removed unless
    # they are the provider default; the code keeps the default in the models
    # list but never emits metadata for it when it is ineligible.
    """Generate complete model inventories and metadata for Tau providers.

    Like Pi, models.dev supplies the model rows while provider transport/auth
    configuration remains application-owned. Existing catalog-only rows are
    retained as explicit Tau corrections; source rows that are deprecated or do
    not advertise tool calling are removed unless they are the provider default.
    """
    if not isinstance(source, Mapping):
        raise ValueError("models.dev data must be a JSON object")

    providers: dict[str, Any] = {}
    for provider in sorted(catalog, key=lambda item: item.name):
        source_key = _MODELS_DEV_PROVIDER_KEYS.get(provider.name, provider.name)
        source_provider = source.get(source_key)
        if not isinstance(source_provider, Mapping):
            continue
        source_models = source_provider.get("models")
        if not isinstance(source_models, Mapping):
            continue

        allowed_models = (provider_model_filters or {}).get(provider.name)
        eligible = {
            model_id: model
            for model_id, model in source_models.items()
            if isinstance(model_id, str)
            and isinstance(model, Mapping)
            and (allowed_models is None or model_id in allowed_models)
            and _is_eligible_model(model)
        }
        source_ids = {model_id for model_id in source_models if isinstance(model_id, str)}
        retained_manual = [
            model
            for model in provider.models
            if model not in source_ids or model == provider.default_model
        ]
        existing_eligible = [model for model in provider.models if model in eligible]
        new_models = sorted(set(eligible) - set(provider.models))
        models = list(dict.fromkeys([*existing_eligible, *retained_manual, *new_models]))
        if not models:
            continue

        metadata = {
            model_id: _model_metadata(model_id, model, provider)
            for model_id, model in sorted(eligible.items())
        }
        providers[provider.name] = {
            "models": models,
            "model_metadata": metadata,
        }

    return {
        "schema_version": 1,
        "source": MODELS_DEV_URL,
        "generated_at": generated_at,
        "providers": providers,
    }


def bundled_models_dev_catalog_document() -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # Loads the generated catalog JSON shipped as a tau_coding package resource,
    # validates it through models_dev_catalog_overlay and returns it if it is a
    # dictionary.
    # Outputs: the document, or none on any read, decode or validation error
    # (errors are swallowed deliberately so Tau falls back to static data).
    # Side effects: reads a package resource file.
    """Load and validate generated model data, or fall back silently."""
    try:
        text = files("tau_coding").joinpath(MODELS_DEV_CATALOG_RESOURCE).read_text(encoding="utf-8")
        document = json.loads(text)
        models_dev_catalog_overlay(document)
        return document if isinstance(document, dict) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def bundled_models_dev_catalog_overlay() -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # Loads the bundled document and returns its raw provider overlay (see
    # models_dev_catalog_overlay), or none when the bundled document is unusable.
    # Side effects: reads a package resource. Does not raise for bad bundled data.
    document = bundled_models_dev_catalog_document()
    return models_dev_catalog_overlay(document) if document is not None else None


def models_dev_catalog_overlay(document: object) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Validates a generated catalog document and converts it to raw partial
    # provider tables for the provider catalog loader.
    # Inputs: the document.
    # Outputs: a mapping with schema_version 1 and a "providers" list of entries
    # each holding name, a copied models list and a copied metadata mapping.
    # Raises ValueError when: schema_version is not 1 or document not a mapping;
    # providers is not a mapping; a provider name is empty/non-string; a provider
    # entry is not a mapping; models is not a non-empty list of unique non-empty
    # strings; metadata is not a mapping; metadata names an unknown model.
    """Validate a generated document and return raw partial provider tables."""
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ValueError("models.dev catalog has an unsupported schema")
    providers = document.get("providers")
    if not isinstance(providers, Mapping):
        raise ValueError("models.dev catalog providers must be an object")

    raw_providers: list[dict[str, Any]] = []
    for provider_name, provider in providers.items():
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError("models.dev provider names must be non-empty strings")
        if not isinstance(provider, Mapping):
            raise ValueError("models.dev provider entries must be objects")
        models = provider.get("models")
        metadata = provider.get("model_metadata")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(model, str) and model for model in models)
            or len(models) != len(set(models))
            or not isinstance(metadata, Mapping)
        ):
            raise ValueError("models.dev provider model data is invalid")
        if any(model not in models for model in metadata):
            raise ValueError("models.dev metadata references an unknown model")
        raw_providers.append(
            {
                "name": provider_name,
                "models": list(models),
                "model_metadata": dict(metadata),
            }
        )
    return {"schema_version": 1, "providers": raw_providers}


def _is_eligible_model(model: Mapping[object, object]) -> bool:
    # [FUNCTION DOC]
    # HELPER: True when a models.dev row advertises tool calling (exactly true),
    # is not deprecated, and, where modality lists exist, includes "text" in both
    # input and output modalities. Rows without modalities are eligible.
    if model.get("tool_call") is not True or model.get("status") == "deprecated":
        return False
    modalities = model.get("modalities")
    if not isinstance(modalities, Mapping):
        return True
    inputs = modalities.get("input")
    outputs = modalities.get("output")
    return not (
        isinstance(inputs, list)
        and "text" not in inputs
        or isinstance(outputs, list)
        and "text" not in outputs
    )


def _model_metadata(
    model_id: str,
    model: Mapping[object, object],
    provider: ProviderCatalogEntry,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds Tau metadata for one eligible model: display name (falls back
    # to id), reasoning flag, input modalities (text plus image when listed), flat
    # cost rates, context window and max output tokens (positive ints, default
    # 4096) and optional context-size cost tiers. When the provider accepts direct
    # effort values (OpenAI-compatible with a reasoning_effort style parameter, or
    # Anthropic with an existing forceAdaptiveThinking compat flag) it adds a
    # thinking_level_map of supported levels and a list of unsupported levels
    # derived from reasoning_options. Pure; returns the metadata mapping.
    modalities = model.get("modalities")
    inputs = modalities.get("input") if isinstance(modalities, Mapping) else None
    cost = model.get("cost")
    limits = model.get("limit")
    metadata: dict[str, Any] = {
        "name": _string_or(model.get("name"), model_id),
        "reasoning": model.get("reasoning") is True,
        "input": ["text", "image"] if isinstance(inputs, list) and "image" in inputs else ["text"],
        "cost": _cost_rates(cost),
        "context_window": _positive_int_or(
            limits.get("context") if isinstance(limits, Mapping) else None, 4096
        ),
        "max_tokens": _positive_int_or(
            limits.get("output") if isinstance(limits, Mapping) else None, 4096
        ),
    }
    cost_tiers = _cost_tiers(cost)
    if cost_tiers:
        metadata["cost_tiers"] = cost_tiers

    existing = provider.model_metadata.get(model_id)
    supports_direct_effort = (
        provider.kind == "openai-compatible"
        and provider.thinking_parameter in {"reasoning_effort", "reasoning.effort"}
        or provider.kind == "anthropic"
        and existing is not None
        and existing.compat.get("forceAdaptiveThinking") is True
    )
    if supports_direct_effort:
        level_map = thinking_level_map_from_reasoning_options(model.get("reasoning_options"))
        if level_map is not None:
            mapped = {level: value for level, value in level_map.items() if value is not None}
            unsupported = [level for level, value in level_map.items() if value is None]
            if mapped:
                metadata["thinking_level_map"] = mapped
            if unsupported:
                metadata["unsupported_thinking_levels"] = unsupported
    return metadata


def _cost_rates(value: object) -> dict[str, float]:
    # [FUNCTION DOC]
    # HELPER: Extracts per-million-token rates (input, output, cacheRead from
    # cache_read, cacheWrite from cache_write) from a cost mapping, using zero for
    # missing/non-numeric values or a non-mapping input.
    cost = value if isinstance(value, Mapping) else {}
    return {
        "input": _number_or_zero(cost.get("input")),
        "output": _number_or_zero(cost.get("output")),
        "cacheRead": _number_or_zero(cost.get("cache_read")),
        "cacheWrite": _number_or_zero(cost.get("cache_write")),
    }


def _cost_tiers(value: object) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Converts models.dev context-size cost tiers into Tau's tier list.
    # Keeps tiers whose descriptor type is "context" with a positive int size,
    # sorted by size. Output: first entry = base rates capped at the smallest tier
    # size, then each tier's rates capped at the next tier's size (the last tier
    # uncapped). Returns empty list when there are no valid tiers.
    if not isinstance(value, Mapping):
        return []
    raw_tiers = value.get("tiers")
    if not isinstance(raw_tiers, list):
        return []
    tiers: list[tuple[int, dict[str, float]]] = []
    for raw_tier in raw_tiers:
        if not isinstance(raw_tier, Mapping):
            continue
        tier = raw_tier.get("tier")
        if not isinstance(tier, Mapping) or tier.get("type") != "context":
            continue
        size = tier.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            continue
        tiers.append((size, _cost_rates(raw_tier)))
    tiers.sort(key=lambda item: item[0])
    if not tiers:
        return []

    output: list[dict[str, Any]] = [{"max_input_tokens": tiers[0][0], **_cost_rates(value)}]
    for index, (_, rates) in enumerate(tiers):
        next_limit = tiers[index + 1][0] if index + 1 < len(tiers) else None
        output.append(
            {
                **({"max_input_tokens": next_limit} if next_limit is not None else {}),
                **rates,
            }
        )
    return output


def _string_or(value: object, fallback: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the value if it is a non-empty string, else the fallback.
    return value if isinstance(value, str) and value else fallback


def _number_or_zero(value: object) -> float:
    # [FUNCTION DOC]
    # HELPER: Returns the value as a float if it is an int/float (not bool), else
    # zero.
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _positive_int_or(value: object, fallback: int) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns the value if it is a positive int (not bool), else fallback.
    return (
        value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback
    )
