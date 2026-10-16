# [MODULE DOC]
# Hugging Face GGUF discovery used by the llama.cpp backend so users can pick
# models for llama.cpp's server-side downloads. Validates exact
# owner/repository[:quantization] references, discovers a search-only HF
# token (env var or standard token files), searches the HF models API for GGUF
# repositories, and summarizes each repository's quantization variants with
# total sizes. Errors are surfaced as secret-free HuggingFaceSearchError.
# Side effects: HTTP requests to huggingface.co and token file reads.
"""Hugging Face GGUF discovery for llama.cpp server-side downloads."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

HF_API_ROOT = "https://huggingface.co/api"
HF_TOKEN_ENV = "HF_TOKEN"
_QUANTIZATION = re.compile(
    r"(?:^|[-_.])((?:UD-)?(?:IQ\d(?:_[A-Z0-9]+)+|Q\d(?:_[A-Z0-9]+)+|"
    r"BF16|F16|F32|MXFP\d(?:_[A-Z0-9]+)*))(?=[-_.]|$)",
    re.IGNORECASE,
)
_EXACT_REPOSITORY = re.compile(r"^[^\s/:]+/[^\s/:]+(?::[^\s:]+)?$")


class HuggingFaceSearchError(RuntimeError):
    # [CLASS DOC]
    # Exception (RuntimeError) for Hugging Face search/details failures; messages
    # are written to never contain tokens or other secrets.
    """A secret-free Hugging Face search/details failure."""


@dataclass(frozen=True, slots=True)
class GgufVariant:
    # [CLASS DOC]
    # Frozen dataclass: one quantization label (upper-case, e.g. Q4_K_M) and its
    # total size in bytes across matching shard files, or None when unknown.
    quantization: str
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class GgufRepository:
    # [CLASS DOC]
    # Frozen dataclass: exact repository id (owner/name), whether access is gated,
    # and its GGUF variants sorted by quantization label.
    id: str
    gated: bool
    variants: tuple[GgufVariant, ...]


def validate_repository_reference(value: str) -> str:
    # [FUNCTION DOC]
    # Strip whitespace and check the value is exactly owner/repository with an
    # optional ":quantization" suffix (no spaces, no extra slashes/colons).
    # Output: the normalized reference. Raises HuggingFaceSearchError otherwise.
    """Return one exact owner/repository[:quantization] download reference."""
    normalized = value.strip()
    if not _EXACT_REPOSITORY.fullmatch(normalized):
        raise HuggingFaceSearchError(
            "Enter an exact Hugging Face owner/repository[:quantization] reference."
        )
    return normalized


def discover_hf_token(
    environment: Mapping[str, str],
    *,
    token_paths: Sequence[Path] | None = None,
) -> str | None:
    # [FUNCTION DOC]
    # Find a Hugging Face token for search. Inputs: environment mapping and
    # optional explicit token file paths. Returns the stripped HF_TOKEN env value
    # if non-empty; else reads each candidate file (explicit paths or the
    # standard locations) and returns the first non-empty stripped content;
    # unreadable/undecodable files are skipped. Returns None if nothing found.
    # Side effects: file reads.
    """Read a search-only token from environment or standard HF token files."""
    token = environment.get(HF_TOKEN_ENV, "").strip()
    if token:
        return token
    paths = tuple(token_paths) if token_paths is not None else _standard_token_paths(environment)
    for path in paths:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if value:
            return value
    return None


async def search_gguf_repositories(
    client: httpx.AsyncClient,
    query: str,
    *,
    token: str | None,
    limit: int = 10,
) -> tuple[GgufRepository, ...]:
    # [FUNCTION DOC]
    # Search GGUF repositories. Inputs: async HTTP client, query, token (may be
    # None), result limit (default 10). If the stripped query is an exact
    # repository reference, any ":quantization" suffix is dropped and a single
    # details lookup is returned. Otherwise GETs the models API filtered to gguf,
    # sorted by downloads descending, then fetches details for each result whose
    # id (or model id) has exactly one slash, sequentially.
    # Output: tuple of GgufRepository. Raises HuggingFaceSearchError for HTTP
    # errors, malformed JSON, or a non-list payload. Network I/O.
    """Search repositories, or inspect one exact owner/repository reference."""
    normalized = query.strip()
    exact = normalized.rsplit(":", 1)[0] if _EXACT_REPOSITORY.fullmatch(normalized) else None
    if exact is not None:
        return (await repository_details(client, exact, token=token),)
    response = await client.get(
        HF_API_ROOT + "/models",
        params={
            "search": normalized,
            "filter": "gguf",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(limit),
        },
        headers=_headers(token),
    )
    _raise_http(response, "searching Hugging Face")
    payload = _json(response)
    if not isinstance(payload, list):
        raise HuggingFaceSearchError("Hugging Face search returned malformed results.")
    repositories: list[GgufRepository] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            continue
        repo_id = raw.get("id", raw.get("modelId"))
        if not isinstance(repo_id, str) or repo_id.count("/") != 1:
            continue
        repositories.append(await repository_details(client, repo_id, token=token))
    return tuple(repositories)


async def repository_details(
    client: httpx.AsyncClient,
    repository: str,
    *,
    token: str | None,
) -> GgufRepository:
    # [FUNCTION DOC]
    # Fetch one repository's metadata and summarize GGUF variants.
    # Inputs: client, repository id, token. Validates the payload is an object
    # with an exact id (falls back to the requested id). Gated is true when the
    # field is literally true or a string other than empty/"false"/"none"
    # (case-insensitive). For each sibling file ending in .gguf (excluding
    # mmproj projector files) whose name contains a recognized quantization label
    # (Q/IQ k-quants, BF16/F16/F32, MXFP, optional UD- prefix), sums file sizes
    # per upper-cased label; if any shard size is unknown the total becomes None.
    # Output: GgufRepository with variants sorted by label.
    # Raises HuggingFaceSearchError on HTTP errors or malformed data. Network I/O.
    response = await client.get(
        f"{HF_API_ROOT}/models/{repository}",
        headers=_headers(token),
    )
    _raise_http(response, f"reading Hugging Face repository {repository}")
    payload = _json(response)
    if not isinstance(payload, Mapping):
        raise HuggingFaceSearchError("Hugging Face repository details were malformed.")
    repo_id = payload.get("id", payload.get("modelId", repository))
    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise HuggingFaceSearchError("Hugging Face repository details omitted an exact id.")
    gated_raw = payload.get("gated", False)
    gated = gated_raw is True or (
        isinstance(gated_raw, str) and gated_raw.casefold() not in {"", "false", "none"}
    )
    totals: dict[str, int | None] = {}
    siblings = payload.get("siblings", ())
    if isinstance(siblings, list):
        for sibling in siblings:
            if not isinstance(sibling, Mapping):
                continue
            filename = sibling.get("rfilename", sibling.get("path"))
            if not isinstance(filename, str) or not filename.casefold().endswith(".gguf"):
                continue
            name = Path(filename).name
            if name.casefold().startswith("mmproj"):
                continue
            match = _QUANTIZATION.search(name)
            if match is None:
                continue
            quantization = match.group(1).upper()
            size = _file_size(sibling)
            prior = totals.get(quantization)
            totals[quantization] = (
                None
                if size is None or (quantization in totals and prior is None)
                else (prior or 0) + size
            )
    variants = tuple(
        GgufVariant(quantization, size)
        for quantization, size in sorted(totals.items(), key=lambda item: item[0])
    )
    return GgufRepository(repo_id, gated, variants)


def _file_size(raw: Mapping[str, object]) -> int | None:
    # [FUNCTION DOC]
    # HELPER: return a sibling's size from its size field or, failing that, its
    # LFS size; only non-negative non-bool ints count, else None.
    size = raw.get("size")
    lfs = raw.get("lfs")
    if not isinstance(size, int) and isinstance(lfs, Mapping):
        size = lfs.get("size")
    return size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None


def _headers(token: str | None) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: return a bearer Authorization header mapping when a token is
    # given, else an empty mapping.
    return {"Authorization": f"Bearer {token}"} if token else {}


def _json(response: httpx.Response) -> object:
    # [FUNCTION DOC]
    # HELPER: parse a response's JSON; raise HuggingFaceSearchError on decode
    # failure.
    try:
        return response.json()
    except ValueError as exc:
        raise HuggingFaceSearchError("Hugging Face returned malformed JSON.") from exc


def _raise_http(response: httpx.Response, operation: str) -> None:
    # [FUNCTION DOC]
    # HELPER: map HTTP failures to HuggingFaceSearchError: 401/403 -> access
    # denied guidance about accepting terms and HF_TOKEN; 429 -> rate limited;
    # other >= 400 -> generic message naming the status and operation. Returns
    # nothing on success.
    if response.status_code in {401, 403}:
        raise HuggingFaceSearchError(
            "Hugging Face denied access. Accept the repository terms and provide HF_TOKEN "
            "for Tau search; the independent llama.cpp server also needs its own HF_TOKEN "
            "for gated downloads."
        )
    if response.status_code == 429:
        raise HuggingFaceSearchError("Hugging Face rate limited the search. Wait and retry.")
    if response.status_code >= 400:
        raise HuggingFaceSearchError(
            f"Hugging Face returned HTTP {response.status_code} while {operation}."
        )


def _standard_token_paths(environment: Mapping[str, str]) -> tuple[Path, ...]:
    # [FUNCTION DOC]
    # HELPER: list candidate HF token files in priority order, de-duplicated:
    # HF_TOKEN_PATH, HF_HOME/token, XDG_CACHE_HOME/huggingface/token,
    # HOME/.cache/huggingface/token, HOME/.huggingface/token (HOME from the
    # mapping, else the real home directory). Empty env values are skipped.
    home = Path(environment.get("HOME") or Path.home())
    explicit = environment.get("HF_TOKEN_PATH", "").strip()
    hf_home = environment.get("HF_HOME", "").strip()
    xdg_cache = environment.get("XDG_CACHE_HOME", "").strip()
    candidates = (
        Path(explicit) if explicit else None,
        Path(hf_home) / "token" if hf_home else None,
        Path(xdg_cache) / "huggingface" / "token" if xdg_cache else None,
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    )
    return tuple(dict.fromkeys(path for path in candidates if path is not None))


__all__ = [
    "GgufRepository",
    "GgufVariant",
    "HF_API_ROOT",
    "HF_TOKEN_ENV",
    "HuggingFaceSearchError",
    "discover_hf_token",
    "repository_details",
    "search_gguf_repositories",
    "validate_repository_reference",
]
