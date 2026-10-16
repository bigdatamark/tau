# [MODULE DOC]
# Built-in coding tools (read, write, edit, bash) for Tau coding sessions, in the
# tau_coding layer so the portable agent core stays free of filesystem and shell
# behavior. Each tool has a ToolDefinition (name, description, prompt snippet and
# guidelines, JSON input schema, async executor taking arguments and an optional
# cancellation token) that converts to a provider-neutral AgentTool. Paths are
# resolved against a working directory fixed at factory time. Text output is
# limited to 2,000 lines or 50 KB (head-truncated for read, tail-truncated for
# bash). Writes and edits are serialized per resolved path with process-wide
# async locks held in a module-level mapping. Also exports text utilities for
# truncation, line-ending handling, exact-replacement edits and diffs.
"""Built-in filesystem and shell tools for Tau coding sessions.

The module exposes factory functions that create provider-neutral `AgentTool`
objects plus richer `ToolDefinition` objects for callers that need prompt
metadata and JSON schemas. The tools operate relative to a configurable working
directory, return structured `AgentToolResult` values, and keep local
filesystem/shell behavior outside the reusable `tau_agent` package.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import signal
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from tau_agent.messages import ImageContent, TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_coding.image_processing import (
    DEFAULT_MAX_SOURCE_IMAGE_BYTES,
    ImageProcessingFailure,
    detect_image_family_mime_type,
    detect_supported_image_mime_type,
    process_image,
    unsupported_image_reason,
)

DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000
IMAGE_SNIFF_BYTES = 64 * 1024
UTF8_BOM = "\ufeff"


class ToolInputError(ValueError):
    # [CLASS DOC]
    # Exception (a ValueError subclass) raised by tool executors for invalid
    # arguments or unusable targets; the agent loop reports it to the model as an
    # error tool result.
    """Raised when a tool receives invalid structured arguments."""


@dataclass(slots=True)
class ImageSupportState:
    # [CLASS DOC]
    # Mutable shared flag describing whether the active model accepts images:
    # None (unknown, images sent), True, or False (read tool omits images). Shared
    # by reference so a model switch updates already-created tools.
    """Mutable active-model image capability shared with built-in tools."""

    supported: bool | None = None


@dataclass(frozen=True, slots=True)
class ReadOperations:
    # [CLASS DOC]
    # Immutable bundle of pluggable filesystem functions used by the read tool:
    # validate a path (raise for unusable paths), read all bytes, and optionally get
    # the size and read a byte prefix (both required to enable the large-image
    # pre-check). Allows reading from non-local backends.
    """Pluggable filesystem operations used by the read tool."""

    validate_path: Callable[[Path], None]
    read_bytes: Callable[[Path], bytes]
    size_bytes: Callable[[Path], int] | None = None
    read_prefix: Callable[[Path, int], bytes] | None = None


@dataclass(frozen=True, slots=True)
class TruncationResult:
    # [CLASS DOC]
    # Immutable description of a truncated output: returned content, whether and
    # why ("lines" or "bytes") it was truncated, total and returned line/byte
    # counts, whether the kept last line is partial (tail mode), whether the first
    # line alone exceeded the byte limit (head mode), and the limits used.
    """Metadata describing how a tool output was shortened.

    `content` contains the returned slice. The remaining fields record whether
    truncation happened, whether the line or byte limit was responsible, the
    total size of the original output, the size of the returned output, and
    edge cases such as partial-line output or a first line that is too large to
    display safely.
    """

    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def to_json(self) -> dict[str, JSONValue]:
        # [FUNCTION DOC]
        # Return all fields as a plain JSON-compatible mapping. Pure.
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    # [CLASS DOC]
    # Immutable full definition of a coding tool before conversion to the core
    # AgentTool: name, description, prompt snippet, prompt guidelines, JSON input
    # schema, and the async executor.
    """Complete definition for a coding tool before provider conversion.

    A definition contains the tool name, user-facing description, prompt
    metadata, JSON input schema, and async executor. `to_agent_tool()` converts
    it into the smaller `AgentTool` type consumed by the provider-neutral agent
    loop while preserving prompt metadata for clients that render tool guidance.
    """

    name: str
    description: str
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...]
    input_schema: Mapping[str, JSONValue]
    executor: Callable[
        [Mapping[str, JSONValue], ToolCancellationToken | None], Awaitable[AgentToolResult]
    ]

    def to_agent_tool(self) -> AgentTool:
        # [FUNCTION DOC]
        # Convert to an AgentTool whose label is the name, carrying the schema and
        # prompt metadata, with an execute function adapting the core signature to the
        # executor. Output: the AgentTool. Pure.
        """Convert the coding definition to the Pi-compatible core tool."""

        async def execute(
            tool_call_id: str,
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
            on_update: ToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            # [FUNCTION DOC]
            # HELPER: Core execute adapter; ignores the tool-call ID and update callback and
            # awaits the executor with the arguments and cancellation token. Propagates any
            # executor exception.
            del tool_call_id, on_update
            return await self.executor(arguments, signal)

        return AgentTool(
            name=self.name,
            label=self.name,
            description=self.description,
            parameters=self.input_schema,
            execute_fn=execute,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


_file_locks: dict[Path, asyncio.Lock] = {}


def _validate_local_read_path(path: Path) -> None:
    # [FUNCTION DOC]
    # HELPER: Raise ToolInputError if the path does not exist or is a directory.
    # Filesystem reads only.
    if not path.exists():
        raise ToolInputError(f"File not found: {path}")
    if path.is_dir():
        raise ToolInputError(f"Path is a directory: {path}")


def _local_file_size(path: Path) -> int:
    # [FUNCTION DOC]
    # HELPER: Return the file size in bytes from a stat call (may raise OSError).
    return path.stat().st_size


def _local_read_prefix(path: Path, limit: int) -> bytes:
    # [FUNCTION DOC]
    # HELPER: Open the file in binary mode and return up to the given number of
    # leading bytes (may raise OSError).
    with path.open("rb") as file:
        return file.read(limit)


DEFAULT_READ_OPERATIONS = ReadOperations(
    validate_path=_validate_local_read_path,
    read_bytes=Path.read_bytes,
    size_bytes=_local_file_size,
    read_prefix=_local_read_prefix,
)


def create_coding_tools(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
    image_support: ImageSupportState | None = None,
) -> list[AgentTool]:
    # [FUNCTION DOC]
    # Create the default tool list in order read, write, edit, bash, rooted at the
    # given cwd (default: current process directory at call time), passing the
    # shared image support state to read and the optional shell prefix to bash.
    # Output: list of AgentTools. No I/O.
    """Create the default coding-tool set for a local project.

    The returned tools are ordered as `read`, `write`, `edit`, and `bash`.
    Relative paths used with those tools are resolved against `cwd`; when `cwd`
    is omitted, the process current working directory at factory-call time is
    used. The tools share per-path write/edit locks within this process so
    concurrent mutations of the same file do not interleave. When configured,
    `shell_command_prefix` is prepended to every bash tool command.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    return [
        create_read_tool(cwd=root, image_support=image_support),
        create_write_tool(cwd=root),
        create_edit_tool(cwd=root),
        create_bash_tool(cwd=root, shell_command_prefix=shell_command_prefix),
    ]


def create_read_tool_definition(
    *,
    cwd: str | Path | None = None,
    operations: ReadOperations | None = None,
    image_support: ImageSupportState | None = None,
) -> ToolDefinition:
    # [FUNCTION DOC]
    # Create the "read" tool definition. Inputs: cwd, optional custom read
    # operations (default local filesystem) and optional image support state.
    # Schema: path (required string), offset and limit (integers). Output: the
    # definition. See the nested executor for behavior.
    # KNOWN ISSUE: the docstring says metadata is returned in "data", but results
    # use the details field; it also lists only ToolInputError while undecodable
    # non-image text raises a Unicode decode error instead.
    """Create a definition for the `read` tool.

    The tool reads a file resolved relative to `cwd` unless an absolute path is
    supplied. Text files are decoded as UTF-8 and may be sliced with optional
    1-indexed `offset` and positive integer `limit` arguments. Returned text is
    truncated to `DEFAULT_MAX_OUTPUT_LINES` lines or `DEFAULT_MAX_OUTPUT_BYTES`
    bytes, whichever comes first, and continuation hints are appended when more
    lines remain. Supported images (`jpg`, `png`, `gif`, `webp`, and `bmp`) are
    detected from file content and returned as provider-neutral image blocks.
    Images are validated and resized or converted when needed to fit inline limits.

    The executor raises `ToolInputError` for invalid arguments, missing files,
    directories, and offsets beyond the end of the file. Successful results
    include the resolved path and truncation metadata in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    read_operations = operations or DEFAULT_READ_OPERATIONS

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Read tool executor.
        # Validates path type, offset at least 0 and limit at least 1 (ToolInputError
        # otherwise), then validates the path via the operations. If size and prefix
        # functions exist and the file is larger than the image source limit (50 MB),
        # sniffs a 64 KB prefix; an image-family file is rejected with an omitted-image
        # result (unsupported variant reason or a size message) without a full read.
        # Otherwise reads all bytes: known-unsupported image variants (animated PNG,
        # JPEG XL) give an omitted-image result; supported images give an omitted
        # result when the model is known not to support images, else are processed in a
        # worker thread (resize/convert); a processing failure yields text "Image
        # omitted: ..." with basic details; success yields a summary text (with notes
        # such as resize/convert) plus a base64 image content item and details (path,
        # source and final MIME types, byte sizes, dimensions).
        # Text files: decoded as UTF-8, line endings normalized to LF, split into lines;
        # offset is 1-based (0 treated as 1) and an offset beyond the line count raises
        # ToolInputError; an optional limit selects lines; the selection is
        # head-truncated. Output text is: a hint to use bash when the first line exceeds
        # the byte limit; or the truncated content plus a "Showing lines a-b of n"
        # continuation hint; or, when a user limit stopped early, the content plus "N
        # more lines in file. Use offset=K to continue."; or the plain content.
        # Details contain the path and the truncation record. The cancellation token is
        # ignored. Side effects: file reads.
        del signal
        raw_path = _str_arg(arguments, "path")
        path = _path_arg(arguments, "path", cwd=root)
        offset = _optional_int_arg(arguments, "offset")
        limit = _optional_int_arg(arguments, "limit")

        if offset is not None and offset < 0:
            raise ToolInputError("offset must be at least 0")
        if limit is not None and limit < 1:
            raise ToolInputError("limit must be at least 1")
        read_operations.validate_path(path)
        if read_operations.size_bytes is not None and read_operations.read_prefix is not None:
            source_size = read_operations.size_bytes(path)
            if source_size > DEFAULT_MAX_SOURCE_IMAGE_BYTES:
                prefix = read_operations.read_prefix(path, IMAGE_SNIFF_BYTES)
                image_family = detect_image_family_mime_type(prefix)
                if image_family is not None:
                    reason = unsupported_image_reason(prefix) or (
                        f"source is {format_size(source_size)}, exceeding the "
                        f"{format_size(DEFAULT_MAX_SOURCE_IMAGE_BYTES)} processing limit"
                    )
                    return _omitted_image_result(
                        path=path,
                        source_mime_type=image_family,
                        source_bytes=source_size,
                        reason=reason,
                    )

        data = read_operations.read_bytes(path)
        unsupported_reason = unsupported_image_reason(data)
        if unsupported_reason is not None:
            image_family = detect_image_family_mime_type(data)
            assert image_family is not None
            return _omitted_image_result(
                path=path,
                source_mime_type=image_family,
                source_bytes=len(data),
                reason=unsupported_reason,
            )

        source_mime_type = detect_supported_image_mime_type(data)
        if source_mime_type is not None:
            if image_support is not None and image_support.supported is False:
                return _omitted_image_result(
                    path=path,
                    source_mime_type=source_mime_type,
                    source_bytes=len(data),
                    reason=(
                        "current model does not support image input. Image contents are "
                        "unavailable; do not infer or describe them. Ask the user to switch "
                        "to a vision-capable model"
                    ),
                )
            processed = await asyncio.to_thread(process_image, data, source_mime_type)
            image_details: dict[str, JSONValue] = {
                "path": str(path),
                "source_mime_type": source_mime_type,
                "bytes": len(data),
            }
            if isinstance(processed, ImageProcessingFailure):
                return AgentToolResult(
                    content=[
                        TextContent(
                            text=(
                                f"Read image file [{source_mime_type}]\n"
                                f"[Image omitted: {processed.message}.]"
                            )
                        )
                    ],
                    details=image_details,
                )

            image_details.update(
                {
                    "mime_type": processed.mime_type,
                    "processed_bytes": len(processed.data),
                    "width": processed.width,
                    "height": processed.height,
                }
            )
            note_lines = "".join(f"\n[{note}]" for note in processed.notes)
            return AgentToolResult(
                content=[
                    TextContent(text=f"Read image file [{processed.mime_type}]{note_lines}"),
                    ImageContent(data=_base64_text(processed.data), mime_type=processed.mime_type),
                ],
                details=image_details,
            )

        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        all_lines = text.split("\n")
        start_line = 0 if offset is None or offset == 0 else offset - 1
        if start_line >= len(all_lines):
            raise ToolInputError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        user_limited_lines: int | None = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected)
        start_display = start_line + 1
        details: dict[str, JSONValue] = {"path": str(path), "truncation": truncation.to_json()}

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode()))
            output = (
                f"[Line {start_display} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit. Use bash: sed -n "
                f"'{start_display}p' {raw_path} | head -c {DEFAULT_MAX_OUTPUT_BYTES}]"
            )
        elif truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            output = truncation.content
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output = (
                f"{truncation.content}\n\n[{remaining} more lines in file. "
                f"Use offset={next_offset} to continue.]"
            )
        else:
            output = truncation.content

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details,
        )

    return ToolDefinition(
        name="read",
        description=(
            "Read the contents of a file. Supports text files and images "
            "(jpg, png, gif, webp, bmp). "
            "Images are sent to vision-capable models as attachments. For text files, output is "
            "truncated to "
            f"{DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB "
            "(whichever is hit first). Use offset/limit for large files. When you need the "
            "full file, continue with offset until complete."
        ),
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of cat or sed.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "Line number to start reading from"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        executor=execute,
    )


def _omitted_image_result(
    *,
    path: Path,
    source_mime_type: str,
    source_bytes: int,
    reason: str,
) -> AgentToolResult:
    # [FUNCTION DOC]
    # HELPER: Build a tool result with text "Read image file [mime]" followed by an
    # "Image omitted: <reason>." line, and details with path, source MIME type and
    # byte count. Pure.
    return AgentToolResult(
        content=[
            TextContent(text=f"Read image file [{source_mime_type}]\n[Image omitted: {reason}.]")
        ],
        details={
            "path": str(path),
            "source_mime_type": source_mime_type,
            "bytes": source_bytes,
        },
    )


def create_read_tool(
    *,
    cwd: str | Path | None = None,
    operations: ReadOperations | None = None,
    image_support: ImageSupportState | None = None,
) -> AgentTool:
    # [FUNCTION DOC]
    # Create the read tool as an AgentTool (definition converted). No I/O.
    """Create an `AgentTool` for reading UTF-8 text files and supported images."""
    return create_read_tool_definition(
        cwd=cwd,
        operations=operations,
        image_support=image_support,
    ).to_agent_tool()


def create_write_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    # [FUNCTION DOC]
    # Create the "write" tool definition rooted at cwd. Schema: path and content
    # (both required strings). See the nested executor.
    # KNOWN ISSUE: the docstring says metadata is returned in "data"; results use
    # the details field.
    """Create a definition for the `write` tool.

    The tool writes the supplied string `content` to `path`, resolving relative
    paths against `cwd`. Parent directories are created automatically and any
    existing file is overwritten. Writes use UTF-8 text encoding and are guarded
    by a per-path async lock so multiple writes/edits to the same resolved file
    are serialized within this process.

    The executor raises `ToolInputError` when `path` or `content` has the wrong
    type. Successful results include the resolved path and number of characters
    written in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Write tool executor. Validates path and content are strings
        # (ToolInputError), then under the per-path lock creates parent directories and
        # writes the content as UTF-8, overwriting any existing file. Output: a result
        # "Successfully wrote to <path>." with details path and character count.
        # Ignores cancellation. Side effects: filesystem writes; OSError propagates.
        del signal
        path = _path_arg(arguments, "path", cwd=root)
        content = _str_arg(arguments, "content")

        async with _file_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        return AgentToolResult(
            content=[TextContent(text=f"Successfully wrote to {path}.")],
            details={"path": str(path), "characters": len(content)},
        )

    return ToolDefinition(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        executor=execute,
    )


def create_write_tool(*, cwd: str | Path | None = None) -> AgentTool:
    # [FUNCTION DOC]
    # Create the write tool as an AgentTool. No I/O.
    """Create an `AgentTool` for creating or overwriting UTF-8 text files."""
    return create_write_tool_definition(cwd=cwd).to_agent_tool()


def create_edit_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    # [FUNCTION DOC]
    # Create the "edit" tool definition rooted at cwd with schema path plus a
    # required non-empty list of oldText/newText objects, and four prompt
    # guidelines. See the nested executor.
    # KNOWN ISSUE: the docstring says metadata is returned in "data"; results use
    # the details field.
    """Create a definition for the `edit` tool.

    The tool applies one or more exact text replacements to a single UTF-8 file
    resolved relative to `cwd`. Each edit item contains `oldText` and `newText`.
    Every `oldText` must be non-empty, must occur exactly once in the original
    file, and must not overlap another edit span. All replacements are validated
    before writing, so the file is left unchanged if any edit fails.

    File content and edit text are normalized to LF for matching, then the
    original file's dominant line ending is restored after replacement. UTF-8
    byte-order marks are preserved. The executor also accepts legacy top-level
    `oldText`/`newText` arguments and JSON-string `edits` values by normalizing
    them into the canonical edits list.

    Successful results include the resolved path, edit count, an ndiff-style
    diff, a unified patch, and the first changed line in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Edit tool executor. Normalizes legacy arguments (JSON-string edits,
        # top-level oldText/newText), validates path and edits, raises ToolInputError if
        # the file is missing or a directory. Under the per-path lock: reads UTF-8 text,
        # strips and remembers a BOM, detects the line ending, normalizes to LF, applies
        # all edits atomically (all validated before any write), restores the line
        # ending and BOM, and writes the file. Output: "Successfully replaced N
        # block(s) in <path>." with details path, edit count, an ndiff-style diff, a
        # unified patch and the first changed line (in LF-normalized content). Raises
        # ToolInputError for invalid edits (empty, not found, not unique, overlapping,
        # no change), leaving the file unchanged. Ignores cancellation. Side effects:
        # file read and write.
        del signal
        prepared = _prepare_edit_arguments(arguments)
        path = _path_arg(prepared, "path", cwd=root)
        edits = _edits_arg(prepared)

        if not path.exists():
            raise ToolInputError(f"Could not edit file: {path}. File not found.")
        if path.is_dir():
            raise ToolInputError(f"Could not edit file: {path}. Path is a directory.")

        async with _file_lock(path):
            raw_content = path.read_text(encoding="utf-8")
            bom, content = _strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)
            base_content, new_content = apply_edits_to_normalized_content(
                normalized, edits, str(path)
            )
            final_content = bom + restore_line_endings(new_content, original_ending)
            path.write_text(final_content, encoding="utf-8")

        diff_text, first_changed_line = generate_diff_string(base_content, new_content)
        patch = generate_unified_patch(str(path), base_content, new_content)
        return AgentToolResult(
            content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
            details={
                "path": str(path),
                "edits": len(edits),
                "diff": diff_text,
                "patch": patch,
                "first_changed_line": first_changed_line,
            },
        )

    return ToolDefinition(
        name="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match "
            "a unique, non-overlapping region of the original file. If two changes affect the "
            "same block or nearby lines, merge them into one edit instead of emitting overlapping "
            "edits. Do not include large unchanged regions just to connect distant changes."
        ),
        prompt_snippet=(
            "Make precise file edits with exact text replacement, including multiple disjoint "
            "edits in one call"
        ),
        prompt_guidelines=(
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with "
            "multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier "
            "edits are applied. Do not emit overlapping or nested edits. Merge nearby "
            "changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. "
            "Do not pad with large unchanged regions.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
        executor=execute,
    )


def create_edit_tool(*, cwd: str | Path | None = None) -> AgentTool:
    # [FUNCTION DOC]
    # Create the edit tool as an AgentTool. No I/O.
    """Create an `AgentTool` for exact, validated text replacement in one file."""
    return create_edit_tool_definition(cwd=cwd).to_agent_tool()


def create_bash_tool_definition(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
) -> ToolDefinition:
    # [FUNCTION DOC]
    # Create the "bash" tool definition rooted at cwd. An optional shell command
    # prefix is stripped (empty becomes none) and prepended on its own line to every
    # command. Schema: command and description required strings (description is
    # display-only and not enforced), optional numeric timeout. See the executor.
    # KNOWN ISSUE: the docstring says metadata is reported in "data"; results use
    # the details field.
    """Create a definition for the `bash` tool.

    The tool runs a shell command with `cwd` as the subprocess working
    directory, disconnects stdin, and combines stdout and stderr into one UTF-8
    decoded output stream. Disconnecting stdin keeps interactive programs from
    competing with Tau for terminal input. The optional `timeout` argument must
    be positive when supplied. On timeout, POSIX commands are started in a new
    session and the entire process group is killed so shell children from
    pipelines or compound commands do not continue running; non-POSIX platforms
    fall back to killing the direct subprocess.

    Output is tail-truncated to `DEFAULT_MAX_OUTPUT_LINES` lines or
    `DEFAULT_MAX_OUTPUT_BYTES` bytes. When truncation occurs, the full output is
    written to a temporary log file and that path is reported in `data`.
    Successful and failed command results both include exit code, timeout state,
    duration, truncation metadata, and full-output path metadata.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    prefix = shell_command_prefix.strip() if shell_command_prefix else None

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Bash tool executor. Validates the command string and a positive
        # timeout (ToolInputError); raises ToolInputError("Command cancelled") if
        # already cancelled. Spawns the command via the shell with cwd, stdin at the
        # null device and stderr merged into stdout; on POSIX in a new session (so its
        # process group can be killed) and using bash explicitly when a prefix is set.
        # Waits for completion, timeout or cancellation (killing the process tree in
        # the latter cases). Decodes output as UTF-8 with replacement, tail-truncates
        # it, and when truncated writes the full output to a temp file and appends a
        # "Showing ... Full output: <path>" notice. Empty output becomes "(no output)".
        # Appends a status block for timeout, cancellation or a non-zero exit code.
        # Output: result text and details (command, exit code, timed_out, cancelled,
        # duration in seconds rounded to ms, truncation record, full output path,
        # whether a prefix was applied). Side effects: process execution, temp files.
        command = _str_arg(arguments, "command")
        shell_command = _prefixed_shell_command(command, prefix)
        timeout = _optional_float_arg(arguments, "timeout")
        if timeout is not None and timeout <= 0:
            raise ToolInputError("timeout must be greater than 0")
        if signal is not None and signal.is_cancelled():
            raise ToolInputError("Command cancelled")

        start = monotonic()
        if os.name == "posix":
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                executable="bash" if prefix else None,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        output_bytes, _stderr, timed_out, cancelled = await _communicate_with_cancellation(
            process,
            timeout=timeout,
            signal=signal,
        )

        output = output_bytes.decode(errors="replace")
        truncation = truncate_tail(output)
        full_output_path: str | None = None
        output_text = truncation.content or "(no output)"
        if truncation.truncated:
            full_output_path = _write_temp_output(output)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. "
                    f"Full output: {full_output_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {full_output_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Full output: {full_output_path}]"
                )

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = (
                f"Command timed out after {timeout:g} seconds" if timeout else "Command timed out"
            )
        elif cancelled:
            status = "Command cancelled"
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status:
            output_text = append_status_block(output_text, status)

        return AgentToolResult(
            content=[TextContent(text=output_text)],
            details={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "duration_seconds": round(monotonic() - start, 3),
                "truncation": truncation.to_json(),
                "full_output_path": full_output_path,
                "shell_command_prefix_applied": prefix is not None,
            },
        )

    return ToolDefinition(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_OUTPUT_LINES} lines or "
            f"{DEFAULT_MAX_OUTPUT_BYTES // 1024}KB (whichever is hit first). If truncated, "
            "full output is saved to a temp file. Optionally provide a timeout in seconds."
        ),
        prompt_snippet="Execute bash commands (ls, grep, find, etc.)",
        prompt_guidelines=(
            "When using bash, include a brief present-participle description of the "
            "command's purpose (for example, 'Running tests').",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "description": {
                    "type": "string",
                    "description": (
                        "Brief present-participle summary of the command's purpose, such as "
                        "'Running tests' or 'Validating and committing changes'"
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)",
                },
            },
            "required": ["command", "description"],
        },
        executor=execute,
    )


def create_bash_tool(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
) -> AgentTool:
    # [FUNCTION DOC]
    # Create the bash tool as an AgentTool. No I/O.
    """Create an `AgentTool` for executing shell commands with captured output."""
    return create_bash_tool_definition(
        cwd=cwd,
        shell_command_prefix=shell_command_prefix,
    ).to_agent_tool()


def _prefixed_shell_command(command: str, prefix: str | None) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the command unchanged without a prefix, else the prefix and the
    # command joined by a newline. Pure.
    """Return a shell command with an opt-in setup prefix applied."""
    if prefix is None:
        return command
    return f"{prefix}\n{command}"


def format_size(bytes_count: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Format a byte count as "<n>B" under 1 KB, "<x.x>KB" under 1 MB, else
    # "<x.x>MB" (1024-based). Pure.
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def append_status_block(text: str, status: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Append the status after a blank line when text is non-empty, else
    # return the status alone. Pure.
    """Append command status text after a blank line when output already exists."""
    return f"{text}\n\n{status}" if text else status


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None,
    signal: ToolCancellationToken | None,
) -> tuple[bytes, bytes | None, bool, bool]:
    # [FUNCTION DOC]
    # HELPER: Collect a subprocess's output while honoring a timeout and a
    # cancellation token. Races the communicate task against an optional
    # cancellation watcher with the timeout. If the process finishes first, returns
    # output, stderr, not timed out, not cancelled. Otherwise kills the process
    # tree, awaits remaining output (empty if that task was cancelled) and returns
    # it with timed_out true unless cancellation fired (then cancelled true).
    # If the surrounding task is cancelled, kills the process tree, cancels the
    # communicate task and re-raises. Always cancels the watcher.
    # Output: a 4-tuple (output bytes, stderr or None, timed out, cancelled).
    communicate = asyncio.create_task(process.communicate())
    cancel_watch: asyncio.Task[None] | None = None
    try:
        wait_for: set[asyncio.Task[Any]] = {communicate}
        if signal is not None:
            cancel_watch = asyncio.create_task(_wait_for_cancel(signal))
            wait_for.add(cancel_watch)

        done, _pending = await asyncio.wait(
            wait_for,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if communicate in done:
            output_bytes, stderr = communicate.result()
            return output_bytes, stderr, False, False

        cancelled = cancel_watch is not None and cancel_watch in done
        _kill_process_tree(process)
        try:
            output_bytes, stderr = await communicate
        except asyncio.CancelledError:
            output_bytes = b""
            stderr_result: bytes | None = None
        else:
            stderr_result = stderr
        return output_bytes, stderr_result, not cancelled, cancelled
    except asyncio.CancelledError:
        _kill_process_tree(process)
        if not communicate.done():
            communicate.cancel()
        raise
    finally:
        if cancel_watch is not None:
            cancel_watch.cancel()


async def _wait_for_cancel(signal: ToolCancellationToken) -> None:
    # [FUNCTION DOC]
    # HELPER: Poll the cancellation token every 50 ms and return once it reports
    # cancelled.
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    # [FUNCTION DOC]
    # Keep the beginning of content within line and byte limits (defaults 2,000
    # lines, 50 KB). Lines are counted without a trailing empty line after a final
    # newline. Unchanged content is returned untruncated. If the first line alone
    # exceeds the byte budget, returns empty content flagged first-line-exceeds.
    # Otherwise keeps whole lines until the line limit (reason "lines") or until the
    # next line (plus its joining newline) would exceed the byte limit (reason
    # "bytes"). Output: a TruncationResult (the recorded limits are always the
    # module defaults). Pure.
    # KNOWN ISSUE: the result always records the default max lines/bytes even when
    # custom limits were passed.
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _truncation_result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    first_line_bytes = len(lines[0].encode()) if lines else 0
    if first_line_bytes > max_bytes:
        return _truncation_result(
            "", True, "bytes", total_lines, total_bytes, 0, 0, first_line=True
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = len(line.encode()) + (1 if index > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return _truncation_result(
        output,
        True,
        truncated_by,
        total_lines,
        total_bytes,
        len(output_lines),
        len(output.encode()),
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    # [FUNCTION DOC]
    # Keep the end of content within line and byte limits (defaults 2,000 lines,
    # 50 KB), walking backward and keeping whole lines. If even the last line does
    # not fit, keeps its trailing bytes only (decoded leniently) and marks the last
    # line as partial. Output: a TruncationResult. Pure.
    # KNOWN ISSUE: as with truncate_head, the recorded limits are always the module
    # defaults rather than the arguments.
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _truncation_result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    last_line_partial = False
    for line in reversed(lines):
        line_bytes = len(line.encode()) + (1 if output_lines else 0)
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                clipped = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, clipped)
                output_bytes = len(clipped.encode())
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return _truncation_result(
        output,
        True,
        truncated_by,
        total_lines,
        total_bytes,
        len(output_lines),
        len(output.encode()),
        last_line_partial=last_line_partial,
    )


def detect_line_ending(content: str) -> str:
    # [FUNCTION DOC]
    # Return "\r\n" when the first line feed in the content is part of a CRLF pair,
    # otherwise "\n" (also when there are no line feeds). Pure.
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    # [FUNCTION DOC]
    # Convert CRLF and lone CR to LF. Pure.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    # [FUNCTION DOC]
    # Convert LF to CRLF when the requested ending is CRLF, else return unchanged.
    # Pure.
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[dict[str, str]],
    path: str,
) -> tuple[str, str]:
    # [FUNCTION DOC]
    # Apply exact replacements to LF-normalized content. Inputs: content, a list of
    # oldText/newText mappings, and the path for messages. Normalizes edit texts to
    # LF; every oldText must be non-empty, occur exactly once (non-overlapping
    # count) in the original content, and the matched spans must not overlap; then
    # replaces from the end backward. Output: a pair (original content, new
    # content). Raises ToolInputError for empty, missing, duplicated, overlapping
    # edits, or when the result is identical to the input. Pure.
    normalized_edits = [
        {"oldText": normalize_to_lf(edit["oldText"]), "newText": normalize_to_lf(edit["newText"])}
        for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if not edit["oldText"]:
            raise ToolInputError(_empty_old_text_error(path, index, len(normalized_edits)))

    matches: list[tuple[int, int, str]] = []
    for index, edit in enumerate(normalized_edits):
        old_text = edit["oldText"]
        occurrences = _count_occurrences(normalized_content, old_text)
        if occurrences == 0:
            raise ToolInputError(_not_found_error(path, index, len(normalized_edits)))
        if occurrences > 1:
            raise ToolInputError(_duplicate_error(path, index, len(normalized_edits), occurrences))
        start = normalized_content.index(old_text)
        matches.append((start, start + len(old_text), edit["newText"]))

    _validate_non_overlapping(matches)
    new_content = normalized_content
    for start, end, new_text in sorted(matches, reverse=True):
        new_content = f"{new_content[:start]}{new_text}{new_content[end:]}"
    if new_content == normalized_content:
        raise ToolInputError(_no_change_error(path, len(normalized_edits)))
    return normalized_content, new_content


def generate_diff_string(old: str, new: str) -> tuple[str, int | None]:
    # [FUNCTION DOC]
    # Produce an ndiff of the two texts by lines and the 1-based line number in the
    # new text of the first change (for a pure deletion, the line after the last
    # kept line); None when there are no changes. Output: a pair (diff text, first
    # changed line). Pure.
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = "\n".join(difflib.ndiff(old_lines, new_lines))
    first_changed_line: int | None = None
    new_line_number = 0
    for line in difflib.ndiff(old_lines, new_lines):
        if line.startswith("  "):
            new_line_number += 1
        elif line.startswith("+"):
            new_line_number += 1
            if first_changed_line is None:
                first_changed_line = new_line_number
        elif line.startswith("-") and first_changed_line is None:
            first_changed_line = max(new_line_number + 1, 1)
    return diff, first_changed_line


def generate_unified_patch(path: str, old: str, new: str) -> str:
    # [FUNCTION DOC]
    # Return a unified diff of old and new texts with the path as both file names
    # (empty string if identical). Pure.
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )


def _truncation_result(
    content: str,
    truncated: bool,
    truncated_by: str | None,
    total_lines: int,
    total_bytes: int,
    output_lines: int,
    output_bytes: int,
    *,
    last_line_partial: bool = False,
    first_line: bool = False,
) -> TruncationResult:
    # [FUNCTION DOC]
    # HELPER: Construct a TruncationResult from positional values plus optional
    # partial-last-line and first-line-exceeds flags, filling the limits with the
    # module defaults. Pure.
    return TruncationResult(
        content=content,
        truncated=truncated,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=first_line,
        max_lines=DEFAULT_MAX_OUTPUT_LINES,
        max_bytes=DEFAULT_MAX_OUTPUT_BYTES,
    )


def _split_lines_for_counting(content: str) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Split content on LF for counting; empty content gives no lines and a
    # trailing newline does not add an extra empty line. Pure.
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the text unchanged if its UTF-8 size fits, else its last
    # max-bytes bytes decoded with invalid partial characters dropped. Pure.
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[-max_bytes:]
    return clipped.decode(errors="ignore")


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the named argument if it is a string, else raise
    # ToolInputError("<name> must be a string").
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a string")
    return value


def _path_arg(arguments: Mapping[str, JSONValue], name: str, *, cwd: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: Read the named string argument, expand a leading "~", and resolve a
    # relative path against cwd (no normalization or existence check). Raises
    # ToolInputError via the string check.
    value = _str_arg(arguments, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _optional_int_arg(arguments: Mapping[str, JSONValue], name: str) -> int | None:
    # [FUNCTION DOC]
    # HELPER: Return None if the argument is missing or null, the value if it is an
    # integer, else raise ToolInputError.
    # KNOWN ISSUE: booleans are accepted as integers.
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ToolInputError(f"{name} must be an integer")
    return value


def _optional_float_arg(arguments: Mapping[str, JSONValue], name: str) -> float | None:
    # [FUNCTION DOC]
    # HELPER: Return None if missing or null, the numeric value as a float if it is
    # an integer or float, else raise ToolInputError. Booleans are accepted.
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ToolInputError(f"{name} must be a number")
    return float(value)


def _prepare_edit_arguments(arguments: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Return a normalized copy of edit arguments: a JSON string "edits"
    # that parses to a list replaces the string; top-level oldText/newText strings
    # are appended as one more edit (to any existing list) and removed. Pure.
    prepared = dict(arguments)
    edits_value = prepared.get("edits")
    if isinstance(edits_value, str):
        try:
            parsed = json.loads(edits_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            prepared["edits"] = parsed

    old_text = prepared.get("oldText")
    new_text = prepared.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = prepared.get("edits")
        edit_list = edits if isinstance(edits, list) else []
        prepared["edits"] = [*edit_list, {"oldText": old_text, "newText": new_text}]
        prepared.pop("oldText", None)
        prepared.pop("newText", None)
    return prepared


def _edits_arg(arguments: Mapping[str, JSONValue]) -> list[dict[str, str]]:
    # [FUNCTION DOC]
    # HELPER: Validate and return the edits as a list of oldText/newText string
    # mappings. Raises ToolInputError when edits is not a non-empty list, an item is
    # not an object, or either text is not a string.
    value = arguments.get("edits")
    if not isinstance(value, list) or not value:
        raise ToolInputError(
            "Edit tool input is invalid. edits must contain at least one replacement."
        )

    edits: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ToolInputError(f"edits[{index}] must be an object")
        old_text = item.get("oldText")
        new_text = item.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolInputError(
                f"edits[{index}].oldText and edits[{index}].newText must be strings"
            )
        edits.append({"oldText": old_text, "newText": new_text})
    return edits


def _validate_non_overlapping(spans: list[tuple[int, int, str]]) -> None:
    # [FUNCTION DOC]
    # HELPER: Sort spans by start and raise ToolInputError("Edits must not
    # overlap") if any span starts before the previous span ends (touching spans are
    # allowed).
    previous_end = -1
    for start, end, _new_text in sorted(spans):
        if start < previous_end:
            raise ToolInputError("Edits must not overlap")
        previous_end = end


def _count_occurrences(content: str, text: str) -> int:
    # [FUNCTION DOC]
    # HELPER: Count non-overlapping occurrences of the text in the content. Pure.
    # KNOWN ISSUE: an empty search text would loop forever (callers reject empty
    # text first).
    count = 0
    start = 0
    while True:
        index = content.find(text, start)
        if index == -1:
            return count
        count += 1
        start = index + len(text)


def _strip_bom(content: str) -> tuple[str, str]:
    # [FUNCTION DOC]
    # HELPER: Return a pair (BOM or empty string, content without the BOM). Pure.
    return (UTF8_BOM, content[1:]) if content.startswith(UTF8_BOM) else ("", content)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Build the "could not find" message, generic for a single edit or
    # naming the edit index for multiple edits. Pure.
    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match exactly "
            "including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly "
        "including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Build the "found N occurrences, must be unique" message, generic for a
    # single edit or naming the edit index. Pure.
    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Build the "oldText must not be empty" message, with the edit index
    # when there are multiple edits. Pure.
    if total_edits == 1:
        return f"oldText must not be empty in {path}."
    return f"edits[{edit_index}].oldText must not be empty in {path}."


def _no_change_error(path: str, total_edits: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Build the "No changes made" message, with extra guidance for a single
    # edit. Pure.
    if total_edits == 1:
        return (
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing "
            "as expected."
        )
    return f"No changes made to {path}. The replacements produced identical content."


def _base64_text(data: bytes) -> str:
    # [FUNCTION DOC]
    # HELPER: Base64-encode bytes and return ASCII text. Pure.
    import base64

    return base64.b64encode(data).decode("ascii")


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    # [FUNCTION DOC]
    # HELPER: On POSIX send SIGKILL to the process group whose ID is the process ID
    # (the process was started in its own session); elsewhere kill only the
    # process. Ignores an already-exited process. Side effect: kills processes.
    if os.name == "posix":
        # `getattr` keeps mypy happy on the Windows stubs (see issue #513).
        killpg = getattr(os, "killpg")  # noqa: B009
        sigkill = getattr(signal, "SIGKILL")  # noqa: B009
        try:
            killpg(process.pid, sigkill)
        except ProcessLookupError:
            return
    else:
        try:
            process.kill()
        except ProcessLookupError:
            return


def _write_temp_output(output: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Write the text to a new persistent temp file named "tau-bash-*.log"
    # (UTF-8) and return its path. Side effect: creates a file that is never
    # deleted.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="tau-bash-",
        suffix=".log",
        delete=False,
    ) as handle:
        handle.write(output)
        return handle.name


class _FileLockContext:
    # [CLASS DOC]
    # HELPER: Async context manager acquiring the process-wide asyncio lock for a
    # resolved path, creating it in the module-level lock mapping on first use.
    # KNOWN ISSUE: locks are never removed from the mapping, so it grows with every
    # distinct path touched.
    def __init__(self, path: Path) -> None:
        # [FUNCTION DOC]
        # HELPER: Resolve the path (symlinks followed) and start with no lock held.
        self._path = path.resolve()
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Get or create the path's lock, remember it and await acquisition.
        lock = _file_locks.setdefault(self._path, asyncio.Lock())
        self._lock = lock
        await lock.acquire()

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Release the acquired lock if any; does not suppress exceptions.
        if self._lock is not None:
            self._lock.release()


def _file_lock(path: Path) -> _FileLockContext:
    # [FUNCTION DOC]
    # HELPER: Return a per-path lock context manager for the given path.
    return _FileLockContext(path)
