# [MODULE DOC]
# Pi-compatible JSONL RPC frontend for a coding session. A client writes one JSON
# command object per line on stdin (with "type" and optional "id"); the server
# writes one compact JSON object per line on stdout: "response" records
# (command, success, optional id, data or error) and, while a prompt runs, the
# session's streamed events. Also projects session entries, trees, model info and
# stats into Pi's camelCase wire shapes. Records larger than 16 MiB are rejected.
"""Pi-compatible JSONL RPC frontend for a Tau coding session."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal, Protocol, cast

import anyio
from pydantic import BaseModel

from tau_agent.messages import AssistantMessage, UserMessage
from tau_agent.session import JsonlSessionStorage
from tau_agent.session.entries import SessionEntry
from tau_agent.types import JSONValue
from tau_coding.commands import CommandRegistry
from tau_coding.events import CodingSessionEvent
from tau_coding.provider_config import (
    AnthropicProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfig,
    provider_thinking_levels,
)
from tau_coding.session import (
    CodingSession,
    ManualCompactionResult,
    ModelChoice,
    TerminalCommandResult,
)
from tau_coding.session_manager import SessionManager
from tau_coding.session_stats import SessionStats

_MAX_RECORD_BYTES = 16 * 1024 * 1024


class RpcSession(Protocol):
    # [CLASS DOC]
    # Structural protocol describing the subset of the coding session used by RPC
    # mode: read-only state (model, provider, thinking levels, model choices,
    # messages, ids/title, session manager, storage, compaction settings, context
    # window/token estimate, queue size, stats, command registry, tree state) and
    # operations (prompt streaming, cancel, model/thinking changes, compaction, new/
    # resume session, entries, tree/branching, export, terminal commands, rename,
    # pending session-start emission and close). Implemented by CodingSession.
    """Public CodingSession surface consumed by RPC mode."""

    @property
    def model(self) -> str:
        # [FUNCTION DOC]
        # Protocol property: the session's current model value. No side effects.
        ...

    @property
    def provider_name(self) -> str:
        # [FUNCTION DOC]
        # Protocol property: the session's current provider name value. No side effects.
        ...

    @property
    def thinking_level(self) -> str:
        # [FUNCTION DOC]
        # Protocol property: the session's current thinking level value. No side effects.
        ...

    @property
    def available_thinking_levels(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Protocol property: the session's current available thinking levels value. No side effects.
        ...

    @property
    def available_model_choices(self) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC]
        # Protocol property: the session's current available model choices value. No side effects.
        ...

    def provider_config(self, provider_name: str) -> ProviderConfig | None:
        # [FUNCTION DOC]
        # Protocol: return the provider configuration for a provider name, or None.
        ...

    @property
    def messages(self) -> tuple[object, ...]:
        # [FUNCTION DOC]
        # Protocol property: the session's current messages value. No side effects.
        ...

    @property
    def session_id(self) -> str | None:
        # [FUNCTION DOC]
        # Protocol property: the session's current session id value. No side effects.
        ...

    @property
    def session_title(self) -> str | None:
        # [FUNCTION DOC]
        # Protocol property: the session's current session title value. No side effects.
        ...

    @property
    def session_manager(self) -> SessionManager | None:
        # [FUNCTION DOC]
        # Protocol property: the session's current session manager value. No side effects.
        ...

    @property
    def storage(self) -> object:
        # [FUNCTION DOC]
        # Protocol property: the session's current storage value. No side effects.
        ...

    @property
    def auto_compact_token_threshold(self) -> int | None:
        # [FUNCTION DOC]
        # Protocol property: the session's current auto compact token threshold value. No side effects.
        ...

    @property
    def auto_compaction_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Protocol property: the session's current auto compaction enabled value. No side effects.
        ...

    @property
    def context_window_tokens(self) -> int:
        # [FUNCTION DOC]
        # Protocol property: the session's current context window tokens value. No side effects.
        ...

    @property
    def context_token_estimate(self) -> int:
        # [FUNCTION DOC]
        # Protocol property: the session's current context token estimate value. No side effects.
        ...

    @property
    def queued_message_count(self) -> int:
        # [FUNCTION DOC]
        # Protocol property: the session's current queued message count value. No side effects.
        ...

    @property
    def session_stats(self) -> SessionStats:
        # [FUNCTION DOC]
        # Protocol property: the session's current session stats value. No side effects.
        ...

    @property
    def command_registry(self) -> CommandRegistry:
        # [FUNCTION DOC]
        # Protocol property: the session's current command registry value. No side effects.
        ...

    @property
    def state(self) -> object:
        # [FUNCTION DOC]
        # Protocol property: the session's current state value. No side effects.
        ...

    def prompt(
        self,
        content: str,
        *,
        streaming_behavior: Literal["steer", "follow_up"] | None = None,
    ) -> AsyncIterator[CodingSessionEvent]:
        # [FUNCTION DOC]
        # Protocol: submit user text; returns an async stream of session events. The
        # keyword streaming behavior (steer or follow_up) selects how to queue the text
        # while the agent is already running.
        ...

    def cancel(self) -> None:
        # [FUNCTION DOC]
        # Protocol: request cancellation of the running agent turn.
        ...

    def set_model_choice(self, choice: ModelChoice) -> None:
        # [FUNCTION DOC]
        # Protocol: switch to the given provider/model choice.
        ...

    async def cycle_thinking_level(self) -> str:
        # [FUNCTION DOC]
        # Protocol: advance to the next thinking level and return it.
        ...

    async def set_thinking_level(self, level: str) -> str:
        # [FUNCTION DOC]
        # Protocol: set a thinking level by name and return the effective level.
        ...

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        # [FUNCTION DOC]
        # Protocol: enable or disable automatic compaction.
        ...

    async def compact_detailed(self, instructions: str | None = None) -> ManualCompactionResult:
        # [FUNCTION DOC]
        # Protocol: compact context now with optional instructions; returns details.
        ...

    async def new_session(self) -> str:
        # [FUNCTION DOC]
        # Protocol: start a fresh session; returns its id.
        ...

    async def resume(self, session_id: str) -> str:
        # [FUNCTION DOC]
        # Protocol: resume a stored session by id; returns the id.
        ...

    async def session_entries(self) -> tuple[SessionEntry, ...]:
        # [FUNCTION DOC]
        # Protocol: return all stored session entries.
        ...

    async def tree_choices(self) -> tuple[object, ...]:
        # [FUNCTION DOC]
        # Protocol: return selectable tree navigation choices.
        ...

    async def branch_to_entry(self, entry_id: str) -> object:
        # [FUNCTION DOC]
        # Protocol: move the active branch to the given entry id.
        ...

    async def export(
        self, destination: Path | None = None, *, format: str | None = None
    ) -> Path:
        # [FUNCTION DOC]
        # Protocol: export the session to an optional destination in an optional format;
        # returns the written path.
        ...

    async def run_terminal_command(
        self, command: str, *, add_to_context: bool
    ) -> TerminalCommandResult:
        # [FUNCTION DOC]
        # Protocol: run a shell command, optionally adding its output to context.
        ...

    async def set_session_name(self, name: str) -> str:
        # [FUNCTION DOC]
        # Protocol: rename the session; returns the stored name.
        ...

    async def emit_pending_session_start(self) -> None:
        # [FUNCTION DOC]
        # Protocol: emit any deferred session-start extension events.
        ...

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Protocol: release session resources.
        ...


class RpcServer:
    # [CLASS DOC]
    # JSONL RPC server bound to one session and a pair of text streams (default
    # process stdin/stdout). Tracks how many prompt streams are active (to report
    # isStreaming and require steer/followUp while busy) and serializes writes with
    # an async lock so concurrent prompt tasks never interleave output lines.
    """Read Pi-style commands and stream responses/events as strict JSONL."""

    def __init__(
        self,
        session: RpcSession,
        *,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the session and streams (defaulting to process stdin/stdout), create the
        # write lock and set the active prompt counter to zero. No I/O.
        self._session = session
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._write_lock = anyio.Lock()
        self._active_prompt_tasks = 0

    async def run(self) -> None:
        # [FUNCTION DOC]
        # Main loop. First emits pending session-start events. Then, inside a task
        # group, repeatedly reads a line from stdin in a worker thread until EOF (empty
        # string): strips one trailing LF and one CR, skips blank lines, rejects lines
        # over 16 MiB (UTF-8) and non-JSON or non-object values with a "parse" error
        # response, and dispatches valid objects. At EOF, if prompts are still running,
        # cancels the session; the task group then waits for prompt tasks to finish.
        # Finally closes the session. Side effects: reads stdin, writes stdout, spawns
        # prompt tasks. Raises whatever escapes prompt tasks or session close.
        """Serve commands until stdin reaches EOF."""
        await self._session.emit_pending_session_start()
        async with anyio.create_task_group() as tasks:
            while True:
                line = await anyio.to_thread.run_sync(self._stdin.readline)
                if line == "":
                    break
                if line.endswith("\n"):
                    line = line[:-1]
                if line.endswith("\r"):
                    line = line[:-1]
                if not line:
                    continue
                if len(line.encode("utf-8")) > _MAX_RECORD_BYTES:
                    await self._error(None, "parse", "RPC record exceeds 16 MiB")
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    await self._error(None, "parse", f"Failed to parse command: {exc.msg}")
                    continue
                if not isinstance(value, dict):
                    await self._error(None, "parse", "Command must be a JSON object")
                    continue
                await self._dispatch(cast(dict[str, object], value), tasks)
            if self._active_prompt_tasks:
                self._session.cancel()
        await self._session.aclose()

    async def _dispatch(self, command: dict[str, object], tasks: anyio.abc.TaskGroup) -> None:
        # [FUNCTION DOC]
        # HELPER: Execute one command object. A missing or non-string "type" yields a
        # "parse" error. Any exception raised while handling is converted into a failed
        # response for that command (message is the exception text), including
        # "Unknown command: <type>". Supported types:
        # prompt / steer / follow_up: requires non-empty "message"; behavior comes from
        # the type or an explicit "streamingBehavior" ("steer" or "followUp"); while a
        # prompt is running a behavior is mandatory. Starts the session prompt, awaits
        # the first event (an immediately empty stream just gets a success response),
        # increments the active count, responds success, then streams the rest in a
        # background task.
        # abort: cancels the session. get_state: model wire info, thinking level,
        # streaming flag, fixed compaction/queue mode values, session file/id/name,
        # auto-compaction flag, message and pending counts. get_messages: all messages.
        # get_available_models: model wire info per choice. set_model: provider
        # (default current) plus required modelId; responds with the new model info.
        # cycle_model: advances to the next choice (wrapping; unknown current starts at
        # the first) or responds with null data when at most one choice exists.
        # cycle_thinking_level: likewise for thinking levels, responding with the level.
        # get_available_thinking_levels; set_thinking_level (required "level");
        # compact (optional customInstructions; responds with summary, first kept id,
        # tokens before/after); set_auto_compaction (required boolean "enabled");
        # bash (required "command", excludeFromContext flag; responds with output and
        # exit code); abort_bash: always an unsupported error; new_session;
        # switch_session (sessionId or sessionPath, resolved via the session manager);
        # get_session_stats; export_html (optional outputPath, home-expanded);
        # get_fork_messages (user message entries as id and text); get_entries
        # (optional "since" cursor: only entries after that id, error if not found;
        # leaf entries omitted; includes the active leaf id); get_tree;
        # get_last_assistant_text (latest assistant message text or null);
        # set_session_name (required "name"); fork (required entryId: branches to it
        # and returns that user message's text or empty string); get_commands
        # (registered commands described as Pi extension commands with a synthetic
        # tau://command/ path).
        # Side effects: session mutations, stdout writes, task spawning.
        request_id = command.get("id")
        command_type = command.get("type")
        if not isinstance(command_type, str):
            await self._error(request_id, "parse", "Command requires a string 'type'")
            return
        try:
            if command_type in {"prompt", "steer", "follow_up"}:
                message = _required_string(command, "message")
                behavior: Literal["steer", "follow_up"] | None = None
                if command_type == "steer":
                    behavior = "steer"
                elif command_type == "follow_up":
                    behavior = "follow_up"
                explicit = command.get("streamingBehavior")
                if explicit is not None:
                    if explicit not in {"steer", "followUp"}:
                        raise ValueError("streamingBehavior must be 'steer' or 'followUp'")
                    behavior = "follow_up" if explicit == "followUp" else "steer"
                if self._active_prompt_tasks and behavior is None:
                    raise ValueError(
                        "Agent is already streaming; set streamingBehavior to steer or followUp"
                    )
                stream = self._session.prompt(message, streaming_behavior=behavior)
                try:
                    first_event = await anext(stream)
                except StopAsyncIteration:
                    await self._response(request_id, command_type)
                    return
                self._active_prompt_tasks += 1
                await self._response(request_id, command_type)
                tasks.start_soon(self._run_prompt, stream, first_event)
                return
            if command_type == "abort":
                self._session.cancel()
                await self._response(request_id, command_type)
                return
            if command_type == "get_state":
                await self._response(
                    request_id,
                    command_type,
                    {
                        "model": _model_wire(self._session),
                        "thinkingLevel": self._session.thinking_level,
                        "isStreaming": self._active_prompt_tasks > 0,
                        "isCompacting": False,
                        "steeringMode": "one-at-a-time",
                        "followUpMode": "one-at-a-time",
                        "sessionFile": _session_file(self._session),
                        "sessionId": self._session.session_id or "",
                        "sessionName": self._session.session_title,
                        "autoCompactionEnabled": self._session.auto_compaction_enabled,
                        "messageCount": len(self._session.messages),
                        "pendingMessageCount": self._session.queued_message_count,
                    },
                )
                return
            if command_type == "get_messages":
                await self._response(
                    request_id, command_type, {"messages": list(self._session.messages)}
                )
                return
            if command_type == "get_available_models":
                await self._response(
                    request_id,
                    command_type,
                    {
                        "models": [
                            _model_wire(self._session, choice=choice)
                            for choice in self._session.available_model_choices
                        ]
                    },
                )
                return
            if command_type == "set_model":
                provider = command.get("provider", self._session.provider_name)
                if not isinstance(provider, str):
                    raise ValueError("provider must be a string")
                self._session.set_model_choice(
                    ModelChoice(
                        provider_name=provider,
                        model=_required_string(command, "modelId"),
                    )
                )
                await self._response(request_id, command_type, _model_wire(self._session))
                return
            if command_type == "cycle_model":
                choices = self._session.available_model_choices
                if len(choices) <= 1:
                    await self._response(request_id, command_type, None, include_data=True)
                    return
                current = ModelChoice(
                    provider_name=self._session.provider_name,
                    model=self._session.model,
                )
                try:
                    index = choices.index(current)
                except ValueError:
                    index = -1
                choice = choices[(index + 1) % len(choices)]
                self._session.set_model_choice(choice)
                await self._response(
                    request_id,
                    command_type,
                    {
                        "model": _model_wire(self._session),
                        "thinkingLevel": self._session.thinking_level,
                        "isScoped": False,
                    },
                )
                return
            if command_type == "cycle_thinking_level":
                levels = self._session.available_thinking_levels
                if len(levels) <= 1:
                    await self._response(request_id, command_type, None, include_data=True)
                    return
                level = await self._session.cycle_thinking_level()
                await self._response(request_id, command_type, {"level": level})
                return
            if command_type == "get_available_thinking_levels":
                await self._response(
                    request_id,
                    command_type,
                    {"levels": list(self._session.available_thinking_levels)},
                )
                return
            if command_type == "set_thinking_level":
                await self._session.set_thinking_level(_required_string(command, "level"))
                await self._response(request_id, command_type)
                return
            if command_type == "compact":
                instructions = _optional_string(command, "customInstructions")
                result = await self._session.compact_detailed(instructions)
                await self._response(
                    request_id,
                    command_type,
                    {
                        "summary": result.summary,
                        "firstKeptEntryId": result.first_kept_entry_id,
                        "tokensBefore": result.tokens_before,
                        "estimatedTokensAfter": result.estimated_tokens_after,
                        "details": {},
                    },
                )
                return
            if command_type == "set_auto_compaction":
                self._session.set_auto_compaction_enabled(_required_bool(command, "enabled"))
                await self._response(request_id, command_type)
                return
            if command_type == "bash":
                bash_result = await self._session.run_terminal_command(
                    _required_string(command, "command"),
                    add_to_context=not bool(command.get("excludeFromContext", False)),
                )
                await self._response(
                    request_id,
                    command_type,
                    {
                        "output": bash_result.output,
                        "exitCode": bash_result.exit_code,
                        "cancelled": False,
                        "truncated": False,
                    },
                )
                return
            if command_type == "abort_bash":
                raise ValueError("abort_bash is not supported by Tau yet")
            if command_type == "new_session":
                await self._session.new_session()
                await self._response(request_id, command_type, {"cancelled": False})
                return
            if command_type == "switch_session":
                session_ref = command.get("sessionId", command.get("sessionPath"))
                if not isinstance(session_ref, str):
                    raise ValueError("switch_session requires sessionPath")
                session_id = _resolve_session_id(self._session, session_ref)
                await self._session.resume(session_id)
                await self._response(request_id, command_type, {"cancelled": False})
                return
            if command_type == "get_session_stats":
                await self._response(
                    request_id,
                    command_type,
                    _session_stats_wire(self._session),
                )
                return
            if command_type == "export_html":
                output_path = _optional_string(command, "outputPath")
                path = await self._session.export(
                    Path(output_path).expanduser() if output_path is not None else None,
                    format="html",
                )
                await self._response(request_id, command_type, {"path": str(path)})
                return
            if command_type == "get_fork_messages":
                entries = await self._session.session_entries()
                await self._response(
                    request_id,
                    command_type,
                    {
                        "messages": [
                            {"entryId": entry.id, "text": entry.message.text}
                            for entry in entries
                            if entry.type == "message" and isinstance(entry.message, UserMessage)
                        ]
                    },
                )
                return
            if command_type == "get_entries":
                cursor_entries = list(await self._session.session_entries())
                since = _optional_string(command, "since")
                if since is not None:
                    try:
                        index = next(
                            i for i, entry in enumerate(cursor_entries) if entry.id == since
                        )
                    except StopIteration as exc:
                        raise ValueError(f"Entry not found: {since}") from exc
                    cursor_entries = cursor_entries[index + 1 :]
                await self._response(
                    request_id,
                    command_type,
                    {
                        "entries": [
                            projected
                            for entry in cursor_entries
                            if (projected := _entry_wire(entry, self._session.provider_name))
                            is not None
                        ],
                        "leafId": _leaf_id(self._session.state),
                    },
                )
                return
            if command_type == "get_tree":
                entries = await self._session.session_entries()
                await self._response(
                    request_id,
                    command_type,
                    {
                        "tree": _tree_wire(entries, self._session.provider_name),
                        "leafId": _leaf_id(self._session.state),
                    },
                )
                return
            if command_type == "get_last_assistant_text":
                text = next(
                    (
                        message.text
                        for message in reversed(self._session.messages)
                        if isinstance(message, AssistantMessage)
                    ),
                    None,
                )
                await self._response(request_id, command_type, {"text": text})
                return
            if command_type == "set_session_name":
                await self._session.set_session_name(_required_string(command, "name"))
                await self._response(request_id, command_type)
                return
            if command_type == "fork":
                entry_id = _required_string(command, "entryId")
                entries = await self._session.session_entries()
                selected_text = next(
                    (
                        entry.message.text
                        for entry in entries
                        if entry.id == entry_id
                        and entry.type == "message"
                        and isinstance(entry.message, UserMessage)
                    ),
                    "",
                )
                await self._session.branch_to_entry(entry_id)
                await self._response(
                    request_id,
                    command_type,
                    {"text": selected_text, "cancelled": False},
                )
                return
            if command_type == "get_commands":
                commands = self._session.command_registry.list_commands()
                await self._response(
                    request_id,
                    command_type,
                    {
                        "commands": [
                            {
                                "name": item.name,
                                "description": item.description,
                                "source": "extension",
                                "sourceInfo": {
                                    "path": "tau://command/" + item.name,
                                    "source": "tau",
                                    "scope": "temporary",
                                    "origin": "top-level",
                                },
                            }
                            for item in commands
                        ]
                    },
                )
                return
            raise ValueError(f"Unknown command: {command_type}")
        except Exception as exc:
            await self._error(request_id, command_type, str(exc))

    async def _run_prompt(
        self,
        stream: AsyncIterator[CodingSessionEvent],
        first_event: CodingSessionEvent,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Background task writing the first event and then each subsequent event
        # of a prompt stream to stdout. RuntimeError or ValueError from the stream is
        # reported as an "rpc_error" record. Always decrements the active prompt count.
        # KNOWN ISSUE: any other exception type escapes, which will cancel the whole task
        # group and terminate the server loop.
        try:
            await self._write(first_event)
            async for event in stream:
                await self._write(event)
        except (RuntimeError, ValueError) as exc:
            await self._write({"type": "rpc_error", "error": str(exc)})
        finally:
            self._active_prompt_tasks -= 1

    async def _response(
        self,
        request_id: object,
        command: str,
        data: object | None = None,
        *,
        include_data: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Write a success response record with type "response", the command
        # name, success true, the request id when not None, and "data" when data is not
        # None or include_data is set (allowing an explicit null).
        response: dict[str, object] = {
            "type": "response",
            "command": command,
            "success": True,
        }
        if request_id is not None:
            response["id"] = request_id
        if data is not None or include_data:
            response["data"] = data
        await self._write(response)

    async def _error(self, request_id: object, command: str, error: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Write a failure response with type "response", the command name,
        # success false, the error message, and the request id when not None.
        response: dict[str, object] = {
            "type": "response",
            "command": command,
            "success": False,
            "error": error,
        }
        if request_id is not None:
            response["id"] = request_id
        await self._write(response)

    async def _write(self, value: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Convert a value to JSON-compatible data, serialize it compactly
        # (no ASCII escaping), then under the write lock write it plus a newline to
        # stdout and flush. Raises on non-serializable data or stream errors.
        payload = json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
        async with self._write_lock:
            self._stdout.write(payload + "\n")
            self._stdout.flush()


def _required_string(command: Mapping[str, object], key: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the command's value for a key if it is a non-empty string;
    # otherwise raise ValueError "<key> must be a non-empty string".
    value = command.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_bool(command: Mapping[str, object], key: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return the command's value for a key if it is a boolean; otherwise
    # raise ValueError "<key> must be a boolean".
    value = command.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_string(command: Mapping[str, object], key: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return None if the key is absent or null, the string if it is a
    # string, else raise ValueError "<key> must be a string". Empty strings allowed.
    value = command.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _leaf_id(state: object) -> object:
    # [FUNCTION DOC]
    # HELPER: Return the state object's active_leaf_id attribute, or None if absent.
    return getattr(state, "active_leaf_id", None)


def _model_wire(session: RpcSession, *, choice: ModelChoice | None = None) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Build Pi's model description for the given choice (default: current
    # provider/model). Looks up provider config and per-model metadata, then
    # derives: api (metadata api, else the configured api of OpenAI-compatible or
    # Anthropic configs, else "openai-responses" for openai-codex, else
    # "openai-completions"); reasoning (metadata flag, else whether the provider
    # offers thinking levels for the model, else whether the session has any);
    # context window (metadata, else provider table, else session window); name
    # (metadata name or model id); baseUrl (metadata, provider, or empty string);
    # input modalities (metadata or ["text"]); maxTokens (metadata or 16384); cost
    # per input/output/cacheRead/cacheWrite defaulting to 0. Returns a mapping.
    selected = choice or ModelChoice(provider_name=session.provider_name, model=session.model)
    provider = session.provider_config(selected.provider_name)
    metadata = provider.model_metadata.get(selected.model) if provider is not None else None
    configured_api = (
        provider.api
        if isinstance(provider, OpenAICompatibleProviderConfig | AnthropicProviderConfig)
        else None
    )
    api = (
        metadata.api
        if metadata is not None and metadata.api is not None
        else configured_api
        if isinstance(configured_api, str)
        else "openai-responses"
        if selected.provider_name == "openai-codex"
        else "openai-completions"
    )
    reasoning = (
        metadata.reasoning
        if metadata is not None and metadata.reasoning is not None
        else bool(provider_thinking_levels(provider, model=selected.model))
        if provider is not None
        else bool(session.available_thinking_levels)
    )
    context_window = (
        metadata.context_window
        if metadata is not None and metadata.context_window is not None
        else provider.context_windows.get(selected.model)
        if provider is not None
        else None
    )
    cost = metadata.cost if metadata is not None else {}
    return {
        "id": selected.model,
        "name": metadata.name if metadata is not None and metadata.name else selected.model,
        "api": api,
        "provider": selected.provider_name,
        "baseUrl": (
            metadata.base_url
            if metadata is not None and metadata.base_url is not None
            else provider.base_url
            if provider is not None
            else ""
        ),
        "reasoning": reasoning,
        "input": list(metadata.input) if metadata is not None and metadata.input else ["text"],
        "contextWindow": context_window or session.context_window_tokens,
        "maxTokens": (
            metadata.max_tokens
            if metadata is not None and metadata.max_tokens is not None
            else 16_384
        ),
        "cost": {
            "input": cost.get("input", 0.0),
            "output": cost.get("output", 0.0),
            "cacheRead": cost.get("cacheRead", 0.0),
            "cacheWrite": cost.get("cacheWrite", 0.0),
        },
    }


def _session_file(session: RpcSession) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the session file path as a string when storage is JSONL file
    # storage, else None.
    storage = session.storage
    return str(storage.path) if isinstance(storage, JsonlSessionStorage) else None


def _session_stats_wire(session: RpcSession) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Build Pi's session stats: session file/id, counts of user, assistant
    # and total messages, tool calls (also reported as tool results), token
    # breakdown (uncached input = input minus cache read minus cache write, floored
    # at 0; output; cache read; cache write; total), estimated cost (0 if unknown)
    # and context usage (estimate, window, percent rounded to 2 decimals).
    # KNOWN ISSUE: a zero context window raises ZeroDivisionError; toolResults simply
    # mirrors the tool call count.
    stats = session.session_stats
    user_messages = sum(isinstance(message, UserMessage) for message in session.messages)
    assistant_messages = sum(isinstance(message, AssistantMessage) for message in session.messages)
    uncached_input = max(
        0,
        stats.input_tokens - stats.cached_input_tokens - stats.cache_write_tokens,
    )
    return {
        "sessionFile": _session_file(session),
        "sessionId": session.session_id or "",
        "userMessages": user_messages,
        "assistantMessages": assistant_messages,
        "toolCalls": stats.tool_call_count,
        "toolResults": stats.tool_call_count,
        "totalMessages": len(session.messages),
        "tokens": {
            "input": uncached_input,
            "output": stats.output_tokens,
            "cacheRead": stats.cached_input_tokens,
            "cacheWrite": stats.cache_write_tokens,
            "total": (
                uncached_input
                + stats.output_tokens
                + stats.cached_input_tokens
                + stats.cache_write_tokens
            ),
        },
        "cost": stats.estimated_cost if stats.estimated_cost is not None else 0.0,
        "contextUsage": {
            "tokens": session.context_token_estimate,
            "contextWindow": session.context_window_tokens,
            "percent": round(
                session.context_token_estimate / session.context_window_tokens * 100,
                2,
            ),
        },
    }


def _entry_wire(entry: SessionEntry, provider_name: str) -> dict[str, JSONValue] | None:
    # [FUNCTION DOC]
    # HELPER: Project one session entry into Pi's entry shape with type, id,
    # parentId and an ISO UTC timestamp ending in "Z". Leaf entries return None.
    # Per type: message (JSON message), custom_message (customType, content,
    # details, display), model_change (provider defaulting to the given provider
    # name, modelId), thinking_level_change (level or "off"), compaction (Pi shape
    # with firstKeptEntryId/tokensBefore/usage/details, or a "custom" entry with
    # customType "tau.compaction" for legacy entries missing those fields),
    # branch_summary (fromId = branch root, else parent, else own id), custom
    # (customType = namespace, data), label (targetId, label), session_info (name =
    # title). Raises AssertionError for an unhandled type.
    if entry.type == "leaf":
        return None
    timestamp = datetime.fromtimestamp(entry.timestamp, tz=UTC)
    base: dict[str, JSONValue] = {
        "type": entry.type,
        "id": entry.id,
        "parentId": entry.parent_id,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
    }
    if entry.type == "message":
        return {**base, "message": _jsonable(entry.message)}
    if entry.type == "custom_message":
        return {
            **base,
            "customType": entry.custom_type,
            "content": _jsonable(entry.content),
            "details": entry.details,
            "display": entry.display,
        }
    if entry.type == "model_change":
        return {
            **base,
            "provider": entry.provider or provider_name,
            "modelId": entry.model,
        }
    if entry.type == "thinking_level_change":
        return {**base, "thinkingLevel": entry.thinking_level or "off"}
    if entry.type == "compaction":
        if entry.first_kept_entry_id is None or entry.tokens_before is None:
            return {
                **base,
                "type": "custom",
                "customType": "tau.compaction",
                "data": {
                    "summary": entry.summary,
                    **({"usage": _jsonable(entry.usage)} if entry.usage is not None else {}),
                },
            }
        return {
            **base,
            "summary": entry.summary,
            "firstKeptEntryId": entry.first_kept_entry_id,
            "tokensBefore": entry.tokens_before,
            **({"usage": _jsonable(entry.usage)} if entry.usage is not None else {}),
            "details": {},
        }
    if entry.type == "branch_summary":
        return {
            **base,
            "fromId": entry.branch_root_id or entry.parent_id or entry.id,
            "summary": entry.summary,
            **({"usage": _jsonable(entry.usage)} if entry.usage is not None else {}),
            "details": {},
        }
    if entry.type == "custom":
        return {**base, "customType": entry.namespace, "data": entry.data}
    if entry.type == "label":
        return {**base, "targetId": entry.target_id, "label": entry.label}
    if entry.type == "session_info":
        return {**base, "name": entry.title}
    raise AssertionError(f"Unhandled Tau session entry: {entry.type}")


def _tree_wire(entries: tuple[SessionEntry, ...], provider_name: str) -> list[JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Build a nested tree of projected entries (leaf entries excluded).
    # Entries whose parent is missing from the set are treated as roots. Returns a
    # list of nodes each with "entry" and "children", preserving entry order.
    # Recursive; very deep trees could hit the recursion limit.
    visible = tuple(entry for entry in entries if entry.type != "leaf")
    children: dict[str | None, list[SessionEntry]] = {}
    ids = {entry.id for entry in visible}
    for entry in visible:
        parent = entry.parent_id if entry.parent_id in ids else None
        children.setdefault(parent, []).append(entry)

    def build(entry: SessionEntry) -> dict[str, JSONValue]:
        # [FUNCTION DOC]
        # HELPER: Recursively project one entry and its children into a node mapping;
        # raises AssertionError if the entry projects to None.
        projected = _entry_wire(entry, provider_name)
        if projected is None:
            raise AssertionError("Leaf entries must be filtered before tree projection")
        return {
            "entry": projected,
            "children": [build(child) for child in children.get(entry.id, [])],
        }

    return [build(entry) for entry in children.get(None, [])]


def _resolve_session_id(session: RpcSession, reference: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Resolve a session reference to an id: raises ValueError if no session
    # manager exists; returns the id directly if the manager knows it; otherwise
    # treats the reference as a path (home-expanded, resolved) and matches it
    # against listed session record paths. Raises ValueError "Unknown session" if
    # nothing matches. Reads the session directory listing.
    manager = session.session_manager
    if manager is None:
        raise ValueError("Session manager is not available")
    direct = manager.get_session(reference)
    if direct is not None:
        return direct.id
    candidate = Path(reference).expanduser().resolve(strict=False)
    for record in manager.list_sessions():
        if record.path.resolve(strict=False) == candidate:
            return record.id
    raise ValueError(f"Unknown session: {reference}")


def _jsonable(value: object) -> JSONValue:
    # [FUNCTION DOC]
    # HELPER: Recursively convert a value to JSON-compatible data: pydantic models
    # via JSON-mode dump by alias; dataclass instances via a recursive field dict;
    # mappings with keys stringified; lists/tuples to lists; primitives as-is;
    # other objects with a model_dump method dumped likewise; everything else via
    # its string form.
    # KNOWN ISSUE: dataclass dumps are not passed back through this function, so
    # nested non-JSON values (paths, datetimes) inside dataclasses can make the
    # later JSON serialization fail.
    if isinstance(value, BaseModel):
        return cast(JSONValue, value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value) and not isinstance(value, type):
        return cast(JSONValue, asdict(value))
    if isinstance(value, Mapping):
        return cast(
            JSONValue,
            {str(key): _jsonable(item) for key, item in value.items()},
        )
    if isinstance(value, (list, tuple)):
        return cast(JSONValue, [_jsonable(item) for item in value])
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return cast(JSONValue, value.model_dump(mode="json", by_alias=True))
    return str(value)


async def run_rpc_session(session: CodingSession) -> None:
    # [FUNCTION DOC]
    # Entry point: run an RPC server over process stdin/stdout for an already
    # configured coding session until stdin closes. Side effects as RpcServer.run.
    """Run RPC mode for an already configured CodingSession."""
    await RpcServer(session).run()
