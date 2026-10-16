# [MODULE DOC]
# Pinned protocol adapter for the llama.cpp server "router" mode (multi-model management)
# used by the built-in llama.cpp extension. Detects router identity and build via the
# props endpoint, gating mutating operations to tested builds b9688..b10595; unknown
# builds fall back to standard OpenAI-compatible discovery in the owning service.
# Also lists router models (state, display name, input modalities, download progress),
# reports modalities from model entries or standard props, streams download progress
# from the router SSE endpoint, and issues load/unload/download requests.
# All network I/O goes through a caller-supplied httpx async client; failures surface as
# LlamaCppRouterError with user-facing messages.
"""Pinned llama.cpp router protocol adapter.

The mutating API is used only after ``/props`` proves both router identity and
a tested build. Unknown builds intentionally fall back to OpenAI-compatible
model discovery in the owning service.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeGuard, cast

import httpx

LLAMA_CPP_ROUTER_MIN_BUILD = 9688
LLAMA_CPP_ROUTER_MAX_BUILD = 10595

InputModalities = tuple[Literal["text", "image"], ...]

# Every input modality llama.cpp can report. A report with any other value is
# not trusted. Tau sends only text and images, so it ignores audio and video.
_KNOWN_INPUT_MODALITIES = frozenset({"text", "image", "audio", "video"})
_SENT_INPUT_MODALITIES = ("text", "image")

RouterState = Literal[
    "loaded", "sleeping", "unloaded", "loading", "downloading", "failed", "unknown"
]


class LlamaCppRouterError(RuntimeError):
    # [CLASS DOC]
    # Runtime error for malformed responses or failed requests to a llama.cpp router,
    # including HTTP errors, auth rejection and malformed JSON. Message is user-facing.
    """A malformed or failed request to a confirmed compatible router."""


@dataclass(frozen=True, slots=True)
class RouterCapability:
    # [CLASS DOC]
    # Immutable result of router detection. role is "standard" (not a router), "compatible"
    # (router with a tested build) or "incompatible" (router with unknown/out-of-range build);
    # optional parsed build number and optional user-facing diagnostic.
    role: Literal["standard", "compatible", "incompatible"]
    build: int | None = None
    diagnostic: str | None = None

    @property
    def compatible(self) -> bool:
        # [FUNCTION DOC]
        # Property: True only when role is "compatible".
        return self.role == "compatible"


@dataclass(frozen=True, slots=True)
class RouterModel:
    # [CLASS DOC]
    # Immutable description of one model listed by the router: exact id, state ("loaded",
    # "sleeping", "unloaded", "loading", "downloading", "failed" or "unknown"), display name,
    # supported input modalities sent by Tau (text/image) or None if unknown, failed flag,
    # and optional aggregate downloaded and total byte counts.
    id: str
    state: RouterState
    display_name: str | None = None
    input_modalities: InputModalities | None = None
    failed: bool = False
    downloaded_bytes: int | None = None
    download_total_bytes: int | None = None


async def detect_router(
    client: httpx.AsyncClient,
    server_root: str,
    headers: Mapping[str, str],
) -> RouterCapability:
    # [FUNCTION DOC]
    # Identify a llama.cpp router and gate it to tested builds.
    # Inputs: async HTTP client, server root URL (no trailing slash expected; "/props" is
    # appended), request headers mapping.
    # Behavior: GET props; 404 -> standard. Other HTTP errors raise via the HTTP check.
    # Parses the JSON object (malformed -> LlamaCppRouterError). If role is not "router" ->
    # standard. Parses build_info of form "b<digits>" optionally followed by "-..." ; missing
    # or unparsable -> incompatible with an "unknown build" diagnostic; build outside the
    # tested range -> incompatible with build and a range diagnostic; else compatible with build.
    # Output: RouterCapability. Raises LlamaCppRouterError, or httpx transport errors.
    """Identify only the documented router and gate it to the tested builds."""
    response = await client.get(server_root + "/props", headers=dict(headers))
    if response.status_code == 404:
        return RouterCapability("standard")
    _raise_http(response, "detecting router capabilities")
    payload = _object(response, "/props")
    if payload.get("role") != "router":
        return RouterCapability("standard")
    build_info = payload.get("build_info")
    match = re.match(r"^b(\d+)(?:-|$)", build_info) if isinstance(build_info, str) else None
    if match is None:
        return RouterCapability(
            "incompatible",
            diagnostic=(
                "Router build is unknown; management is disabled and Tau is using "
                "standard OpenAI-compatible discovery."
            ),
        )
    build = int(match.group(1))
    if not LLAMA_CPP_ROUTER_MIN_BUILD <= build <= LLAMA_CPP_ROUTER_MAX_BUILD:
        return RouterCapability(
            "incompatible",
            build,
            (
                f"Router build b{build} is outside Tau's tested range "
                f"b{LLAMA_CPP_ROUTER_MIN_BUILD}-b{LLAMA_CPP_ROUTER_MAX_BUILD}; "
                "management is disabled."
            ),
        )
    return RouterCapability("compatible", build)


async def list_router_models(
    client: httpx.AsyncClient,
    server_root: str,
    headers: Mapping[str, str],
    *,
    reload: bool = False,
) -> tuple[RouterModel, ...]:
    # [FUNCTION DOC]
    # List models from the router models endpoint.
    # Inputs: client, server root, headers; keyword reload flag adds query reload=1.
    # Raises LlamaCppRouterError for HTTP errors, non-object JSON, a non-list "data" field,
    # non-mapping entries, ids that are not non-empty strings without surrounding whitespace,
    # or duplicate ids. For each entry: status.failed truthy -> state "failed"; else
    # status.value among the known states is used; otherwise "unknown". Display name is
    # "name" (falling back to "display_name") if a non-blank string, else the id. Input
    # modalities come from reported_input_modalities; download progress from the status.
    # Output: tuple of RouterModel in server order. Network I/O only.
    response = await client.get(
        server_root + "/models",
        params={"reload": "1"} if reload else None,
        headers=dict(headers),
    )
    _raise_http(response, "listing router models")
    payload = _object(response, "/models")
    data = payload.get("data")
    if not isinstance(data, list):
        raise LlamaCppRouterError("llama.cpp /models returned a malformed model list.")
    result: list[RouterModel] = []
    ids: set[str] = set()
    for raw in data:
        if not isinstance(raw, Mapping):
            raise LlamaCppRouterError("llama.cpp /models returned a malformed model.")
        model_id = raw.get("id")
        if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
            raise LlamaCppRouterError("llama.cpp /models returned a model without an exact id.")
        if model_id in ids:
            raise LlamaCppRouterError("llama.cpp /models returned duplicate model ids.")
        ids.add(model_id)
        status = raw.get("status")
        value = status.get("value") if isinstance(status, Mapping) else None
        failed = bool(status.get("failed")) if isinstance(status, Mapping) else False
        state: RouterState
        if failed:
            state = "failed"
        elif value in {"loaded", "sleeping", "unloaded", "loading", "downloading"}:
            state = cast(RouterState, value)
        else:
            state = "unknown"
        modalities = reported_input_modalities(raw)
        display = raw.get("name", raw.get("display_name"))
        downloaded_bytes, download_total_bytes = _download_progress(status)
        result.append(
            RouterModel(
                id=model_id,
                state=state,
                display_name=(
                    display if isinstance(display, str) and display.strip() else model_id
                ),
                input_modalities=modalities,
                failed=failed,
                downloaded_bytes=downloaded_bytes,
                download_total_bytes=download_total_bytes,
            )
        )
    return tuple(result)


def reported_input_modalities(model: Mapping[str, object]) -> InputModalities | None:
    # [FUNCTION DOC]
    # Return the text/image input modalities a model entry reports.
    # Input: a model mapping. Looks at architecture.input_modalities first (llama.cpp),
    # otherwise top-level input_modalities, otherwise modalities.
    # Returns None if the value is not a non-empty list, contains anything other than the
    # known strings text/image/audio/video, or yields no text/image entries; otherwise a
    # tuple of the reported entries among text and image, in the order text, image.
    """Return the text and image inputs that a llama.cpp ``/models`` entry reports.

    llama.cpp reports them as ``architecture.input_modalities``. Other servers may
    use a top-level ``input_modalities`` or ``modalities`` list. Returns None when
    the entry reports nothing, or reports a value Tau does not know.
    """
    architecture = model.get("architecture")
    reported = architecture.get("input_modalities") if isinstance(architecture, Mapping) else None
    if reported is None:
        reported = model.get("input_modalities", model.get("modalities"))
    if not isinstance(reported, list) or not reported:
        return None
    if not all(isinstance(entry, str) and entry in _KNOWN_INPUT_MODALITIES for entry in reported):
        return None
    sent = tuple(entry for entry in _SENT_INPUT_MODALITIES if entry in reported)
    return cast(InputModalities, sent) or None


def props_input_modalities(props: object) -> InputModalities | None:
    # [FUNCTION DOC]
    # Derive input modalities from a standard server's props payload: if modalities.vision
    # is a boolean, returns text+image when true or text only when false; otherwise None.
    # Accepts any object; non-mappings yield None.
    """Return the inputs that ``modalities.vision`` in a standard server's ``/props`` reports."""
    modalities = props.get("modalities") if isinstance(props, Mapping) else None
    vision = modalities.get("vision") if isinstance(modalities, Mapping) else None
    if not isinstance(vision, bool):
        return None
    return ("text", "image") if vision else ("text",)


def _download_progress(status: object) -> tuple[int | None, int | None]:
    # [FUNCTION DOC]
    # HELPER: Aggregate download progress from a status mapping's "progress" mapping, whose
    # values are per-file mappings with "done" and "total" numbers. Skips entries whose
    # done/total are not finite non-negative non-boolean numbers. Returns the integer sums
    # (downloaded, total), or (None, None) if status/progress is not a mapping, no valid
    # entries were found, or the total is not positive. Pure function.
    if not isinstance(status, Mapping):
        return None, None
    progress = status.get("progress")
    if not isinstance(progress, Mapping):
        return None, None
    downloaded = 0.0
    total = 0.0
    found = False
    for value in progress.values():
        if not isinstance(value, Mapping):
            continue
        done = value.get("done")
        size = value.get("total")
        if not _is_non_negative_number(done) or not _is_non_negative_number(size):
            continue
        downloaded += float(done)
        total += float(size)
        found = True
    if not found or total <= 0:
        return None, None
    return int(downloaded), int(total)


def _is_non_negative_number(value: object) -> TypeGuard[int | float]:
    # [FUNCTION DOC]
    # HELPER: Type guard: True for finite int/float values greater than or equal to zero,
    # excluding booleans.
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


async def watch_router_download_progress(
    client: httpx.AsyncClient,
    server_root: str,
    headers: Mapping[str, str],
    model_id: str,
    callback: Callable[[int, int], None],
) -> None:
    # [FUNCTION DOC]
    # Stream the router's models SSE endpoint and forward download progress for one model.
    # Inputs: client, server root, headers, model id, and a callback taking (downloaded
    # bytes, total bytes). Opens a GET stream with no timeout; HTTP errors raise via the
    # HTTP check. Accumulates "data:" lines (leading whitespace after the prefix removed)
    # until a blank line ends the event, then forwards the joined data; events whose data
    # exceeds about 1,000,000 characters are dropped. A final unterminated event is also
    # forwarded. Other SSE fields are ignored. Runs until the server closes the stream or
    # the caller cancels. Returns nothing; callback invocations are the side effect.
    # KNOWN ISSUE: on an HTTP error the error-detail helper reads the JSON body of a
    # streamed response that was never read, which in httpx raises a not-read error rather
    # than ValueError, so the caller may get that error instead of LlamaCppRouterError.
    """Forward bounded aggregate download progress from the router SSE stream."""
    async with client.stream(
        "GET",
        server_root + "/models/sse",
        headers=dict(headers),
        timeout=None,
    ) as response:
        _raise_http(response, "watching model progress")
        data_lines: list[str] = []
        data_size = 0
        async for line in response.aiter_lines():
            if line:
                if line.startswith("data:"):
                    value = line[5:].lstrip()
                    data_size += len(value)
                    if data_size <= 1_000_000:
                        data_lines.append(value)
                continue
            if data_lines and data_size <= 1_000_000:
                _forward_download_progress("\n".join(data_lines), model_id, callback)
            data_lines = []
            data_size = 0
        if data_lines and data_size <= 1_000_000:
            _forward_download_progress("\n".join(data_lines), model_id, callback)


def _forward_download_progress(
    data: str,
    model_id: str,
    callback: Callable[[int, int], None],
) -> None:
    # [FUNCTION DOC]
    # HELPER: Parse one SSE data payload as JSON; silently ignore invalid JSON, non-object
    # payloads, events for other models, or event types other than "download_progress".
    # Otherwise aggregates its "data" progress and calls the callback with (downloaded,
    # total) when both are available.
    try:
        event = json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(event, Mapping):
        return
    if event.get("model") != model_id or event.get("event") != "download_progress":
        return
    downloaded, total = _download_progress(event.get("data"))
    if downloaded is not None and total is not None:
        callback(downloaded, total)


async def mutate_router_model(
    client: httpx.AsyncClient,
    server_root: str,
    headers: Mapping[str, str],
    *,
    action: Literal["load", "unload", "download"],
    model_id: str,
) -> None:
    # [FUNCTION DOC]
    # Request a router model action. Keyword inputs: action "load", "unload" or "download"
    # and the exact model id. POSTs JSON with the model id to the models endpoint for
    # download, or to models/<action> otherwise. Raises LlamaCppRouterError on HTTP errors,
    # malformed JSON, or when the response's "success" field is not exactly true.
    # Returns nothing.
    endpoint = "/models" if action == "download" else f"/models/{action}"
    response = await client.post(
        server_root + endpoint,
        headers=dict(headers),
        json={"model": model_id},
    )
    _raise_http(response, f"requesting model {action}")
    payload = _object(response, endpoint)
    if payload.get("success") is not True:
        raise LlamaCppRouterError(f"llama.cpp did not accept the model {action} request.")


def _object(response: httpx.Response, endpoint: str) -> Mapping[str, object]:
    # [FUNCTION DOC]
    # HELPER: Decode a response body as a JSON object. Raises LlamaCppRouterError naming
    # the endpoint when the body is not valid JSON or not a mapping; returns the mapping.
    try:
        payload = response.json()
    except ValueError as exc:
        raise LlamaCppRouterError(f"llama.cpp {endpoint} returned malformed JSON.") from exc
    if not isinstance(payload, Mapping):
        raise LlamaCppRouterError(f"llama.cpp {endpoint} returned malformed JSON.")
    return payload


def _raise_http(response: httpx.Response, operation: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Raise LlamaCppRouterError for HTTP 401/403 (advising to check the API key or
    # LLAMA_API_KEY) or any status at or above 400 (message includes status, the operation
    # description and, if available, the server's error detail). Returns nothing otherwise.
    if response.status_code in {401, 403}:
        raise LlamaCppRouterError(
            "llama.cpp rejected the router request. Check the optional API key or LLAMA_API_KEY."
        )
    if response.status_code >= 400:
        detail = _server_error_detail(response)
        suffix = f": {detail}" if detail else "."
        raise LlamaCppRouterError(
            f"llama.cpp returned HTTP {response.status_code} while {operation}{suffix}"
        )


def _server_error_detail(response: httpx.Response) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Extract error.message from a JSON error body, collapse whitespace and cap it
    # to 300 characters. Returns None if the body is not JSON, not an object, has no string
    # message, or the message is blank.
    """Extract one bounded, user-actionable message from a router error."""
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    message = error.get("message") if isinstance(error, Mapping) else None
    if not isinstance(message, str):
        return None
    normalized = " ".join(message.split())
    if not normalized:
        return None
    return normalized[:300]


__all__ = [
    "LLAMA_CPP_ROUTER_MAX_BUILD",
    "LLAMA_CPP_ROUTER_MIN_BUILD",
    "LlamaCppRouterError",
    "RouterCapability",
    "RouterModel",
    "RouterState",
    "detect_router",
    "list_router_models",
    "mutate_router_model",
    "props_input_modalities",
    "reported_input_modalities",
    "watch_router_download_progress",
]
