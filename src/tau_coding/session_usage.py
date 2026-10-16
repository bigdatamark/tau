# [MODULE DOC]
# Token-usage analytics for session HTML export. Walks session entries to build
# per-request token usage (assistant responses plus compaction and branch
# summary model calls), tool-call counts and notable events (compactions,
# model/thinking changes, branch summaries), estimates USD cost from catalog
# rates, and renders a self-contained dashboard (stat cards, inline SVG line
# charts, request table, tool counts) plus the CSS and JavaScript strings the
# export page embeds for hover tooltips, legend toggling and PNG download.
# Pure string generation; no I/O.
"""Token-usage analytics for Tau sessions, rendered as an HTML dashboard."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from tau_agent.messages import AssistantMessage, Usage
from tau_agent.session import (
    BranchSummaryEntry,
    CompactionEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    ThinkingLevelChangeEntry,
)
from tau_coding.provider_catalog import builtin_provider_entry, model_cost_for_input_tokens
from tau_coding.session_stats import _response_cost
from tau_coding.tui.themes import TAU_DARK_THEME, TAU_LIGHT_THEME

__all__ = [
    "RequestUsage",
    "SessionUsage",
    "UsageEvent",
    "USAGE_SCRIPT",
    "USAGE_STYLES",
    "collect_session_usage",
    "render_usage_dashboard",
]


@dataclass(frozen=True, slots=True)
class RequestUsage:
    # [CLASS DOC]
    # Immutable usage record for one model request: 1-based number, kind
    # (assistant, compaction summary, branch summary), HH:MM:SS UTC time, provider,
    # model, optional upstream response provider, fresh input, cache-read, cache
    # write (including 1-hour writes), 1-hour cache write, output and reasoning
    # token counts, stop reason and optional estimated cost.
    """Token usage for a single assistant response."""

    number: int
    kind: str
    timestamp: str
    provider: str
    model: str
    response_provider: str | None
    fresh: int
    cached: int
    cache_write: int
    cache_write_1h: int
    output: int
    reasoning: int
    stop_reason: str
    estimated_cost: float | None

    @property
    def prompt(self) -> int:
        # [FUNCTION DOC]
        # Property: total prompt tokens, the sum of fresh, cached and cache-write.
        return self.fresh + self.cached + self.cache_write

    @property
    def hit_rate(self) -> float:
        # [FUNCTION DOC]
        # Property: cached divided by prompt tokens, or 0.0 when prompt is zero.
        return self.cached / self.prompt if self.prompt else 0.0


@dataclass(frozen=True, slots=True)
class UsageEvent:
    # [CLASS DOC]
    # Immutable chart event: the request number it is anchored to, time string,
    # kind (compaction, model, thinking, branch), display label and whether it
    # happened before (default) or after that request.
    """A notable session event positioned against the next model request."""

    request_number: int
    timestamp: str
    kind: str
    label: str
    position: Literal["before", "after"] = "before"


@dataclass(frozen=True, slots=True)
class SessionUsage:
    # [CLASS DOC]
    # Immutable aggregate: tuple of requests, tool-call counts as (name, count)
    # pairs sorted by descending count then name, compaction count and events.
    # Exposes derived totals.
    """Aggregated usage for the entries shown in an export."""

    requests: tuple[RequestUsage, ...]
    tool_calls: tuple[tuple[str, int], ...]
    compactions: int
    events: tuple[UsageEvent, ...] = ()

    @property
    def total_fresh(self) -> int:
        # [FUNCTION DOC]
        # Property: sum of fresh input over all requests.
        return sum(item.fresh for item in self.requests)

    @property
    def total_cached(self) -> int:
        # [FUNCTION DOC]
        # Property: sum of cache-read tokens over all requests.
        return sum(item.cached for item in self.requests)

    @property
    def total_cache_write(self) -> int:
        # [FUNCTION DOC]
        # Property: sum of cache-write tokens over all requests.
        return sum(item.cache_write for item in self.requests)

    @property
    def total_prompt(self) -> int:
        # [FUNCTION DOC]
        # Property: total fresh plus cached plus cache-write tokens.
        return self.total_fresh + self.total_cached + self.total_cache_write

    @property
    def total_output(self) -> int:
        # [FUNCTION DOC]
        # Property: sum of output tokens over all requests.
        return sum(item.output for item in self.requests)

    @property
    def hit_rate(self) -> float | None:
        # [FUNCTION DOC]
        # Property: overall cached share of prompt tokens, or none when there are no
        # prompt tokens or no caching activity at all (no reads and no writes).
        if self.total_prompt <= 0:
            return None
        if self.total_cached == 0 and self.total_cache_write == 0:
            return None
        return self.total_cached / self.total_prompt

    @property
    def total_cost(self) -> float | None:
        # [FUNCTION DOC]
        # Property: sum of known request cost estimates, or none if none are known.
        costs = [item.estimated_cost for item in self.requests if item.estimated_cost is not None]
        return sum(costs) if costs else None


def estimated_request_cost(
    provider: str,
    model: str,
    *,
    fresh: int,
    cached: int,
    cache_write: int,
    cache_write_1h: int,
    output: int,
) -> float | None:
    # [FUNCTION DOC]
    # Estimates one request's USD cost from the built-in provider catalog only.
    # Inputs: provider, model and keyword token counts (fresh, cached, cache
    # write including 1-hour writes, 1-hour cache write, output).
    # Output: cost, or none when the provider/model metadata or rates are
    # unavailable. Rates may be tiered by total prompt size; 1-hour writes use the
    # 1-hour rate when present. No side effects.
    """Estimate a request cost in USD from the built-in provider catalog rates."""
    entry = builtin_provider_entry(provider)
    metadata = entry.model_metadata.get(model) if entry is not None else None
    if metadata is None:
        return None
    rates = model_cost_for_input_tokens(metadata, fresh + cached + cache_write)
    if not rates:
        return None
    # cache_write already includes the 1-hour TTL writes; Anthropic bills those at
    # the higher cacheWrite1h rate, falling back to the 5-minute rate when the
    # catalog entry has no cacheWrite1h value.
    return _response_cost(
        input_tokens=fresh,
        output_tokens=output,
        cache_read_tokens=cached,
        cache_write_tokens=cache_write,
        cache_write_1h_tokens=cache_write_1h,
        rates=rates,
    )


def collect_session_usage(entries: Sequence[SessionEntry]) -> SessionUsage:
    # [FUNCTION DOC]
    # Builds a SessionUsage from entries in order. Tracks the current provider and
    # model (updated by model-change entries and assistant messages, both starting
    # as "unknown"). Compaction and branch-summary entries with usage add a
    # request (using their own provider/model or the current ones); compactions
    # are counted. Assistant messages add a request (with stop reason) and count
    # their tool calls by name. Event entries are queued and attached as "before"
    # events to the next request recorded; any left at the end are attached as
    # "after" events of the last request (dropped if there were no requests).
    # When no catalog cost is available but the message reports a positive cost
    # total, that reported cost is used. Output: the aggregate. No side effects.
    """Collect per-request token usage, tool-call counts, and notable events."""
    requests: list[RequestUsage] = []
    tools: dict[str, int] = {}
    compactions = 0
    pending_events: list[tuple[str, str, str]] = []
    events: list[UsageEvent] = []
    current_provider = "unknown"
    current_model = "unknown"

    def append_request(
        usage: Usage,
        *,
        timestamp: float,
        kind: str,
        provider: str,
        model: str,
        response_provider: str | None = None,
        stop_reason: str = "-",
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: closure that creates and appends a RequestUsage numbered after the
        # existing ones (estimating cost, defaulting missing 1-hour writes and
        # reasoning to zero), then flushes queued events onto it. Mutates the
        # enclosing request, event and pending lists.
        cache_write_1h = usage.cache_write_1h or 0
        estimated = estimated_request_cost(
            provider,
            model,
            fresh=usage.input,
            cached=usage.cache_read,
            cache_write=usage.cache_write,
            cache_write_1h=cache_write_1h,
            output=usage.output,
        )
        if estimated is None and usage.cost.total > 0:
            estimated = usage.cost.total
        request_number = len(requests) + 1
        requests.append(
            RequestUsage(
                number=request_number,
                kind=kind,
                timestamp=_entry_time(timestamp),
                provider=provider,
                model=model,
                response_provider=response_provider,
                fresh=usage.input,
                cached=usage.cache_read,
                cache_write=usage.cache_write,
                cache_write_1h=cache_write_1h,
                output=usage.output,
                reasoning=usage.reasoning or 0,
                stop_reason=stop_reason,
                estimated_cost=estimated,
            )
        )
        events.extend(
            UsageEvent(
                request_number=request_number,
                timestamp=event_timestamp,
                kind=event_kind,
                label=label,
            )
            for event_timestamp, event_kind, label in pending_events
        )
        pending_events.clear()

    for entry in entries:
        event = _usage_event(entry)
        if isinstance(entry, CompactionEntry):
            compactions += 1
            if entry.usage is not None:
                append_request(
                    entry.usage,
                    timestamp=entry.timestamp,
                    kind="compaction summary",
                    provider=entry.provider or current_provider,
                    model=entry.model or current_model,
                    response_provider=entry.response_provider,
                )
            if event is not None:
                kind, label = event
                pending_events.append((_entry_time(entry.timestamp), kind, label))
            continue
        if isinstance(entry, BranchSummaryEntry):
            if entry.usage is not None:
                append_request(
                    entry.usage,
                    timestamp=entry.timestamp,
                    kind="branch summary",
                    provider=entry.provider or current_provider,
                    model=entry.model or current_model,
                    response_provider=entry.response_provider,
                )
            if event is not None:
                kind, label = event
                pending_events.append((_entry_time(entry.timestamp), kind, label))
            continue
        if event is not None:
            kind, label = event
            pending_events.append((_entry_time(entry.timestamp), kind, label))
        if isinstance(entry, ModelChangeEntry):
            current_model = entry.model
            if entry.provider is not None:
                current_provider = entry.provider
            continue
        if not isinstance(entry, MessageEntry):
            continue
        message = entry.message
        if not isinstance(message, AssistantMessage):
            continue
        current_provider = message.provider
        current_model = message.model
        for call in message.tool_calls:
            tools[call.name] = tools.get(call.name, 0) + 1
        append_request(
            message.usage,
            timestamp=entry.timestamp,
            kind="assistant",
            provider=message.provider,
            model=message.model,
            response_provider=message.response_provider,
            stop_reason=message.stop_reason,
        )
    if requests:
        events.extend(
            UsageEvent(
                request_number=len(requests),
                timestamp=timestamp,
                kind=kind,
                label=label,
                position="after",
            )
            for timestamp, kind, label in pending_events
        )
    ordered_tools = tuple(sorted(tools.items(), key=lambda item: (-item[1], item[0])))
    return SessionUsage(
        requests=tuple(requests),
        tool_calls=ordered_tools,
        compactions=compactions,
        events=tuple(events),
    )


def _entry_time(timestamp: float) -> str:
    # [FUNCTION DOC]
    # HELPER: formats an epoch-seconds timestamp as HH:MM:SS in UTC.
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%H:%M:%S")


def _usage_event(entry: SessionEntry) -> tuple[str, str] | None:
    # [FUNCTION DOC]
    # Returns (kind, label) chart metadata for compaction, model change ("Model
    # changed to X"), thinking level change ("Thinking changed to X", off when
    # unset) and branch summary entries; none for other entries.
    """Return chart metadata for session events that can affect prompt usage."""
    if isinstance(entry, CompactionEntry):
        return "compaction", "Compaction"
    if isinstance(entry, ModelChangeEntry):
        return "model", f"Model changed to {entry.model}"
    if isinstance(entry, ThinkingLevelChangeEntry):
        return "thinking", f"Thinking changed to {entry.thinking_level or 'off'}"
    if isinstance(entry, BranchSummaryEntry):
        return "branch", "Branch summary"
    return None


_SERIES_COLORS = {
    # Derived from the built-in themes so exported charts cannot drift from them.
    "cached": (TAU_DARK_THEME.accent, TAU_LIGHT_THEME.accent),
    "cache writes": (TAU_DARK_THEME.success, TAU_LIGHT_THEME.success),
    "fresh": (TAU_DARK_THEME.error, TAU_LIGHT_THEME.error),
    "request": (TAU_DARK_THEME.error, TAU_LIGHT_THEME.error),
    "cumulative": (TAU_DARK_THEME.markdown_link, TAU_LIGHT_THEME.markdown_link),
    "output": (
        TAU_DARK_THEME.role_styles["branch_summary"].border,
        TAU_LIGHT_THEME.role_styles["branch_summary"].border,
    ),
    "reasoning": (TAU_DARK_THEME.success, TAU_LIGHT_THEME.success),
    "event": (TAU_DARK_THEME.markdown_bullet, TAU_LIGHT_THEME.markdown_bullet),
}


def _series_color_pair(name: str) -> tuple[str, str]:
    # [FUNCTION DOC]
    # HELPER: returns the (dark, light) theme colors for a chart series name,
    # defaulting to the markdown link colors.
    return _SERIES_COLORS.get(
        name,
        (TAU_DARK_THEME.markdown_link, TAU_LIGHT_THEME.markdown_link),
    )


def _compact_number(value: float) -> str:
    # [FUNCTION DOC]
    # HELPER: formats a number with "m" (one decimal) for millions, "k" (no
    # decimals) for thousands, else as an integer string.
    # KNOWN ISSUE: values just below one million round up to "1000k".
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _format_cost(value: float | None) -> str:
    # [FUNCTION DOC]
    # HELPER: formats a cost as dollars: "$N/A" for none, three decimals for
    # positive values under one cent, otherwise two decimals.
    if value is None:
        return "$N/A"
    if 0 < value < 0.01:
        return f"${value:.3f}"
    return f"${value:.2f}"


def _line_chart(
    title: str,
    series: Sequence[tuple[str, Sequence[float]]],
    *,
    y_max: float | None = None,
    percent: bool = False,
    timestamps: Sequence[str] | None = None,
    events: Sequence[UsageEvent] = (),
) -> str:
    # [FUNCTION DOC]
    # HELPER: renders one line chart as an inline SVG string (900x330 view box).
    # Inputs: title, series as (name, values) pairs, keyword optional fixed y
    # maximum, percent mode, per-request timestamps and events.
    # The y maximum is the given value or the observed max (at least 1), rounded
    # up to a multiple of its leading power of ten unless in percent mode. Draws
    # six grid lines with tick labels, x tick labels (thinned beyond 20 points),
    # a hidden hover line, an event line and marker per event (clamped to the
    # request range, staggered vertically), one polyline and hidden hover point
    # per series with exact value labels in data attributes, clickable legend
    # toggles laid out right to left, and an x-axis label. Both dark and light
    # colors are stored in data attributes for the page script. All text is HTML
    # escaped. Output: SVG markup.
    """Render one interactive line chart as inline SVG."""
    width, height = 900, 330
    left, right, top, bottom = 68, 20, 42, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    count = max((len(values) for _, values in series), default=0)
    observed_max = max((max(values, default=0.0) for _, values in series), default=0.0)
    maximum = y_max or max(observed_max, 1.0)
    if not percent:
        magnitude = 10 ** max(0, len(str(int(maximum))) - 1)
        maximum = ((maximum + magnitude - 1) // magnitude) * magnitude

    def point(index: int, value: float) -> tuple[float, float]:
        # [FUNCTION DOC]
        # HELPER: maps a request index and value to SVG x/y coordinates within the
        # plot area.
        x = left + (index / max(count - 1, 1)) * plot_width
        y = top + plot_height - (value / maximum) * plot_height
        return x, y

    ts_attr = (
        f' data-timestamps="{html.escape("|".join(timestamps), quote=True)}"' if timestamps else ""
    )
    parts = [
        f'<svg class="usage-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title, quote=True)}" data-left="{left}" '
        f'data-right="{right}" data-count="{count}"{ts_attr}>',
        f'<text class="chart-title" x="{left}" y="24">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height - tick * plot_height / 5
        label = f"{value:.0%}" if percent else _compact_number(value)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
            f'<text class="tick" x="{left - 9}" y="{y + 4:.1f}" text-anchor="end">{label}</text>'
        )
    for index in range(count):
        if count <= 20 or index % max(1, count // 12) == 0 or index == count - 1:
            x, _ = point(index, 0)
            parts.append(
                f'<text class="tick" x="{x:.1f}" y="{height - 22}" '
                f'text-anchor="middle">{index + 1}</text>'
            )
    parts.append(
        f'<line class="hover-line" x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}" visibility="hidden"/>'
    )
    event_dark, event_light = _series_color_pair("event")
    for event_index, event in enumerate(events):
        x, _ = point(max(0, min(count - 1, event.request_number - 1)), 0)
        marker_y = top + 7 + (event_index % 3) * 9
        description = (
            f"{event.label} {event.position} request {event.request_number} at {event.timestamp}"
        )
        parts.append(
            f'<g class="usage-event usage-event-{html.escape(event.kind, quote=True)}" '
            f'data-request="{event.request_number}" '
            f'data-event-info="{html.escape(description, quote=True)}" '
            f'role="img" aria-label="{html.escape(description, quote=True)}">'
            f"<title>{html.escape(description)}</title>"
            f'<line class="event-line" data-dark="{event_dark}" data-light="{event_light}" '
            f'x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" '
            f'stroke="{event_dark}"/>'
            f'<circle class="event-marker" data-dark="{event_dark}" data-light="{event_light}" '
            f'cx="{x:.1f}" cy="{marker_y:.1f}" r="4" fill="{event_dark}"/></g>'
        )
    for series_index, (name, values) in enumerate(series):
        dark, light = _series_color_pair(name)
        points = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in (point(i, value) for i, value in enumerate(values))
        )
        labels = "|".join(f"{value:.1%}" if percent else f"{int(value):,}" for value in values)
        parts.append(
            f'<g class="series" data-series-id="{series_index}" '
            f'data-name="{html.escape(name, quote=True)}" '
            f'data-labels="{labels}">'
        )
        parts.append(
            f'<polyline class="series-line" data-dark="{dark}" data-light="{light}" '
            f'points="{points}" fill="none" stroke="{dark}" stroke-width="3"/>'
            f'<circle class="hover-point" data-dark="{dark}" data-light="{light}" '
            f'r="5" fill="{dark}" visibility="hidden"/></g>'
        )
    legend_x = float(width - right)
    for series_index, (name, _) in reversed(list(enumerate(series))):
        dark, light = _series_color_pair(name)
        legend_x -= 18 + len(name) * 8
        parts.append(
            f'<g class="series-toggle" data-series-id="{series_index}" role="button" '
            f'tabindex="0" aria-pressed="true">'
            f'<circle class="series-swatch" data-dark="{dark}" data-light="{light}" '
            f'cx="{legend_x:.1f}" cy="20" r="4" fill="{dark}"/>'
            f'<text class="legend" x="{legend_x + 8:.1f}" y="24">{html.escape(name)}</text></g>'
        )
    parts.append(
        f'<text class="axis-label" x="{left + plot_width / 2}" y="{height - 4}" '
        'text-anchor="middle">Model request</text></svg>'
    )
    return "".join(parts)


def _figure(chart: str) -> str:
    # [FUNCTION DOC]
    # HELPER: wraps chart markup in a figure with a PNG download button.
    return (
        f'<figure class="usage-figure">{chart}'
        '<button type="button" class="png-button" '
        'title="Download this chart as a PNG image">PNG</button></figure>'
    )


def render_usage_dashboard(usage: SessionUsage) -> str:
    # [FUNCTION DOC]
    # Renders the Usage tab markup. With no requests returns an "empty" message
    # paragraph. Otherwise builds nine stat cards (requests, cache hit rate or
    # N/A, cached, cache writes, fresh, total prompt, output, estimated cost,
    # compactions), charts for prompt input (with event markers), cache hit rate
    # per request and cumulative (only when caching occurred) and output vs
    # reasoning, an explanatory note, a per-request table (hit rate shown as N/A
    # when no caching occurred) and tool-call counts, plus a tooltip container.
    # Output: HTML string. No side effects.
    """Render the usage dashboard markup for the export's Usage tab."""
    requests = usage.requests
    if not requests:
        return (
            '<p class="empty">No assistant responses with token usage were found '
            "in this session.</p>"
        )

    timestamps = [item.timestamp for item in requests]
    cached = [float(item.cached) for item in requests]
    cache_writes = [float(item.cache_write) for item in requests]
    fresh = [float(item.fresh) for item in requests]
    outputs = [float(item.output) for item in requests]
    reasoning = [float(item.reasoning) for item in requests]
    rates = [item.hit_rate for item in requests]
    cumulative_rates: list[float] = []
    running_cached = running_prompt = 0
    for item in requests:
        running_cached += item.cached
        running_prompt += item.prompt
        cumulative_rates.append(running_cached / running_prompt if running_prompt else 0.0)
    cache_hit_rate = usage.hit_rate

    cards = [
        ("Model requests", f"{len(requests):,}"),
        ("Cache hit rate", f"{cache_hit_rate:.1%}" if cache_hit_rate is not None else "N/A"),
        ("Cached input", f"{usage.total_cached:,}"),
        ("Cache writes", f"{usage.total_cache_write:,}"),
        ("Fresh input", f"{usage.total_fresh:,}"),
        ("Total prompt input", f"{usage.total_prompt:,}"),
        ("Output tokens", f"{usage.total_output:,}"),
        ("Estimated cost", _format_cost(usage.total_cost)),
        ("Compactions", f"{usage.compactions}"),
    ]
    cards_html = "".join(
        f'<div class="usage-card"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )

    charts = [
        _line_chart(
            "Prompt input by request",
            [("cached", cached), ("cache writes", cache_writes), ("fresh", fresh)],
            timestamps=timestamps,
            events=usage.events,
        )
    ]
    if cache_hit_rate is not None:
        charts.append(
            _line_chart(
                "Cache hit rate",
                [("request", rates), ("cumulative", cumulative_rates)],
                y_max=1.0,
                percent=True,
                timestamps=timestamps,
            )
        )
    charts.append(
        _line_chart(
            "Output and reasoning tokens",
            [("output", outputs), ("reasoning", reasoning)],
            timestamps=timestamps,
        )
    )
    charts_html = "".join(_figure(chart) for chart in charts)
    show_hit_rates = cache_hit_rate is not None

    table_rows = "".join(
        f"<tr><td>{item.number}</td><td>{html.escape(item.kind)}</td>"
        f"<td>{html.escape(item.timestamp)}</td><td>{html.escape(item.provider)}</td>"
        f"<td>{html.escape(item.response_provider) if item.response_provider else '-'}</td>"
        f"<td>{html.escape(item.model)}</td><td>{item.fresh:,}</td><td>{item.cached:,}</td>"
        f"<td>{item.cache_write:,}</td><td>{item.prompt:,}</td>"
        f"<td>{f'{item.hit_rate:.1%}' if show_hit_rates else 'N/A'}</td>"
        f"<td>{item.output:,}</td><td>{_format_cost(item.estimated_cost)}</td>"
        f"<td>{html.escape(item.stop_reason)}</td></tr>"
        for item in requests
    )
    tool_rows = (
        "".join(
            f'<div class="usage-tool"><span>{html.escape(name)}</span>'
            f"<strong>{count}</strong></div>"
            for name, count in usage.tool_calls
        )
        or '<p class="empty">No tool calls.</p>'
    )

    return (
        f'<div class="usage-cards">{cards_html}</div>'
        '<p class="usage-note">Costs use Tau\'s provider catalog rates. OAuth subscription '
        "estimates are API-rate equivalents, not actual subscription charges. Hover a request "
        "for exact values, select a legend item to hide a series, and use PNG to save a "
        "chart. Event markers show compactions, model or thinking changes, and branch summaries."
        "</p>"
        f'<div class="usage-charts">{charts_html}</div>'
        '<div class="usage-details">'
        '<div class="usage-panel"><h2>Requests</h2><div class="usage-table-wrap"><table>'
        "<thead><tr><th>#</th><th>Request</th><th>Time</th><th>Provider</th>"
        "<th>Response Provider</th><th>Model</th><th>Fresh</th>"
        "<th>Cached</th>"
        "<th>Written</th><th>Prompt</th><th>Hit rate</th><th>Output</th><th>Est. cost</th>"
        f"<th>Stop</th></tr></thead><tbody>{table_rows}</tbody></table></div></div>"
        f'<div class="usage-panel"><h2>Tool calls</h2>{tool_rows}</div>'
        "</div>"
        '<div class="usage-tooltip" role="status" aria-live="polite"></div>'
    )


USAGE_STYLES = """
    .usage-cards {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1px;
      background: var(--line);
      border: 1px solid var(--line);
    }
    .usage-card {
      min-width: 0;
      padding: 16px 18px;
      background: var(--surface);
      transition: background .16s ease;
    }
    .usage-card:hover { background: var(--surface-2); }
    .usage-card span {
      display: block;
      color: var(--accent);
      font-size: 0.62rem;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .usage-card strong {
      display: block;
      margin-top: 6px;
      color: var(--bright);
      font-size: 1.25rem;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
    }
    .usage-note {
      margin: 18px 0 0;
      padding: 2px 0 2px 13px;
      border-left: 2px solid var(--line);
      color: var(--muted);
      font-size: 0.74rem;
    }
    .usage-note::before {
      content: "# ";
      color: var(--accent);
      font-weight: 700;
    }
    .usage-charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 18px;
    }
    .usage-figure {
      position: relative;
      margin: 0;
      padding: 16px 16px 6px;
      background: var(--surface);
      border: 1px solid var(--line);
    }
    .usage-figure:first-child { grid-column: 1 / -1; }
    .usage-chart {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
      cursor: crosshair;
      touch-action: none;
    }
    .png-button {
      position: absolute;
      top: 10px;
      right: 10px;
      padding: 3px 9px;
      color: var(--muted);
      background: var(--surface-2);
      border: 1px solid var(--line-strong);
      font-family: var(--mono);
      font-size: 0.62rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      cursor: pointer;
      transition: color .15s, border-color .15s;
    }
    .png-button:hover { color: var(--bright); border-color: var(--accent); }
    .png-button[disabled] { opacity: .5; cursor: progress; }
    .grid { stroke: var(--line-strong); stroke-width: 1; opacity: .75; }
    .chart-title {
      fill: var(--bright);
      font-family: var(--mono);
      font-size: 15px;
      font-weight: 600;
    }
    .tick, .axis-label, .legend {
      fill: var(--muted);
      font-family: var(--mono);
      font-size: 11px;
    }
    .hover-line {
      stroke: var(--accent);
      stroke-width: 1;
      stroke-dasharray: 3 4;
      pointer-events: none;
    }
    .event-line { stroke-width: 1.5; stroke-dasharray: 5 4; opacity: .8; }
    .event-marker { stroke: var(--surface); stroke-width: 2; }
    .usage-event:hover .event-line, .usage-event:hover .event-marker { opacity: 1; }
    .hover-point { stroke: var(--surface); stroke-width: 2; pointer-events: none; }
    .series { transition: opacity .15s ease; }
    .series.is-hidden { opacity: .07; pointer-events: none; }
    .series-toggle { cursor: pointer; outline: none; }
    .series-toggle[aria-pressed="false"] { opacity: .25; }
    .series-toggle:hover text { text-decoration: underline; }
    .usage-tooltip {
      position: fixed;
      z-index: 30;
      display: none;
      min-width: 170px;
      padding: 10px 12px;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--line-strong);
      box-shadow: 0 10px 35px rgba(0, 0, 0, .35);
      font-family: var(--mono);
      font-size: 0.74rem;
      line-height: 1.7;
      white-space: pre-line;
      pointer-events: none;
    }
    .usage-details {
      display: grid;
      grid-template-columns: minmax(0, 3fr) minmax(190px, 1fr);
      gap: 16px;
      margin-top: 16px;
    }
    .usage-panel {
      min-width: 0;
      padding: 18px;
      background: var(--surface);
      border: 1px solid var(--line);
    }
    .usage-panel h2 { margin: 0 0 14px; text-transform: lowercase; }
    .usage-panel h2::before { content: "$ "; color: var(--accent); }
    .usage-table-wrap {
      overflow: auto;
      max-height: 560px;
      border: 1px solid var(--line);
      scrollbar-color: var(--line-strong) var(--surface);
    }
    .usage-panel table {
      border-collapse: collapse;
      width: 100%;
      white-space: nowrap;
      font-size: 0.72rem;
      font-variant-numeric: tabular-nums;
    }
    .usage-panel th, .usage-panel td {
      padding: 8px 10px;
      text-align: right;
      border-bottom: 1px solid var(--line);
    }
    .usage-panel tbody tr { transition: background .12s ease; }
    .usage-panel tbody tr:hover { background: var(--surface-2); color: var(--bright); }
    .usage-panel th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--bg);
      color: var(--muted);
      font-size: 0.58rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border-bottom: 1px solid var(--line-strong);
    }
    .usage-panel th:nth-child(2), .usage-panel th:nth-child(4),
    .usage-panel th:nth-child(5), .usage-panel th:nth-child(6),
    .usage-panel td:nth-child(2), .usage-panel td:nth-child(4),
    .usage-panel td:nth-child(5), .usage-panel td:nth-child(6) { text-align: left; }
    .usage-tool {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 2px;
      border-bottom: 1px solid var(--line);
      font-size: 0.76rem;
    }
    .usage-tool:last-child { border-bottom: none; }
    .usage-tool strong { color: var(--accent); font-weight: 500; }
    @media (max-width: 820px) {
      .usage-cards { grid-template-columns: 1fr 1fr; }
      .usage-charts { grid-template-columns: 1fr; }
      .usage-figure:first-child { grid-column: auto; }
      .usage-details { grid-template-columns: 1fr; }
    }
"""

USAGE_SCRIPT = """
    (function () {
      var tooltip = document.querySelector(".usage-tooltip");
      var charts = Array.prototype.slice.call(document.querySelectorAll(".usage-chart"));

      // The Tau palette ships both variants; switch series colors with the page theme.
      function syncTheme() {
        var dark = document.documentElement.classList.contains("theme-dark");
        charts.forEach(function (chart) {
          var colored = chart.querySelectorAll("[data-dark][data-light]");
          Array.prototype.forEach.call(colored, function (node) {
            var color = dark ? node.dataset.dark : node.dataset.light;
            var tagName = node.tagName.toLowerCase();
            if (tagName === "polyline" || tagName === "line") {
              node.setAttribute("stroke", color);
            } else {
              node.setAttribute("fill", color);
            }
          });
        });
      }
      window.addEventListener("tau-themechange", syncTheme);
      syncTheme();

      function hideTooltip(chart) {
        if (!tooltip) {
          return;
        }
        tooltip.style.display = "none";
        var line = chart.querySelector(".hover-line");
        if (line) {
          line.setAttribute("visibility", "hidden");
        }
        chart.querySelectorAll(".hover-point").forEach(function (point) {
          point.setAttribute("visibility", "hidden");
        });
      }
      charts.forEach(function (chart) {
        var count = Number(chart.dataset.count);
        var left = Number(chart.dataset.left);
        var right = Number(chart.dataset.right);
        var viewWidth = chart.viewBox.baseVal.width;
        chart.addEventListener("pointermove", function (event) {
          var rect = chart.getBoundingClientRect();
          var svgX = (event.clientX - rect.left) * viewWidth / rect.width;
          var plotWidth = viewWidth - left - right;
          if (svgX < left || svgX > viewWidth - right || count < 1) {
            hideTooltip(chart);
            return;
          }
          var ratio = (svgX - left) / plotWidth * Math.max(count - 1, 1);
          var index = Math.max(0, Math.min(count - 1, Math.round(ratio)));
          var activeSeries = Array.prototype.filter.call(
            chart.querySelectorAll(".series"),
            function (series) { return !series.classList.contains("is-hidden"); }
          );
          if (!activeSeries.length) {
            return;
          }
          var tooltipLines = [];
          activeSeries.forEach(function (series) {
            var polyline = series.querySelector(".series-line");
            var point = polyline && polyline.points.getItem(index);
            var marker = series.querySelector(".hover-point");
            if (!point || !marker) {
              return;
            }
            marker.setAttribute("cx", point.x);
            marker.setAttribute("cy", point.y);
            marker.setAttribute("visibility", "visible");
            var labels = (series.dataset.labels || "").split("|");
            tooltipLines.push(series.dataset.name + "  " + labels[index]);
          });
          chart.querySelectorAll('.usage-event[data-request="' + (index + 1) + '"]')
            .forEach(function (sessionEvent) {
              tooltipLines.push("Event  " + sessionEvent.dataset.eventInfo);
            });
          var x = activeSeries[0].querySelector(".hover-point").getAttribute("cx");
          var line = chart.querySelector(".hover-line");
          line.setAttribute("x1", x);
          line.setAttribute("x2", x);
          line.setAttribute("visibility", "visible");
          if (!tooltip) {
            return;
          }
          var ts = chart.dataset.timestamps;
          var tsLabel = ts ? " @ " + ts.split("|")[index] : "";
          tooltip.textContent = "Request " + (index + 1) + tsLabel
            + "\\n" + tooltipLines.join("\\n");
          tooltip.style.display = "block";
          var tooltipRect = tooltip.getBoundingClientRect();
          var maxLeft = window.innerWidth - tooltipRect.width - 10;
          tooltip.style.left = Math.min(maxLeft, event.clientX + 14) + "px";
          tooltip.style.top = Math.max(10, event.clientY - tooltipRect.height - 12) + "px";
        });
        chart.addEventListener("pointerleave", function () {
          hideTooltip(chart);
        });
        function toggleSeries(control) {
          var id = control.dataset.seriesId;
          var series = chart.querySelector('.series[data-series-id="' + id + '"]');
          var visible = control.getAttribute("aria-pressed") === "true";
          control.setAttribute("aria-pressed", String(!visible));
          series.classList.toggle("is-hidden", visible);
        }
        chart.querySelectorAll(".series-toggle").forEach(function (control) {
          control.addEventListener("click", function (event) {
            event.stopPropagation();
            toggleSeries(control);
          });
          control.addEventListener("keydown", function (event) {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              toggleSeries(control);
            }
          });
        });
      });

      var SVG_NS = "http:" + "//www.w3.org/2000/svg";
      // The light-theme variants are designed to remain legible on white.
      var PRINT_CSS = [
        'text{font-family:"JetBrains Mono",Consolas,Menlo,monospace}',
        ".chart-title{fill:#111827;font-size:15px;font-weight:600}",
        ".tick,.axis-label,.legend{fill:#475569;font-size:11px}",
        ".grid{stroke:#cbd5e1;stroke-width:1}",
        ".point{stroke:#ffffff;stroke-width:1}"
      ].join("");

      function staticSvg(chart) {
        var width = chart.viewBox.baseVal.width;
        var height = chart.viewBox.baseVal.height;
        var clone = chart.cloneNode(true);
        clone.setAttribute("xmlns", SVG_NS);
        clone.setAttribute("width", width);
        clone.setAttribute("height", height);
        clone.removeAttribute("class");
        clone.querySelectorAll(".hover-line").forEach(function (node) {
          node.remove();
        });
        clone.querySelectorAll(".series.is-hidden").forEach(function (node) {
          var id = node.dataset.seriesId;
          var control = clone.querySelector('.series-toggle[data-series-id="' + id + '"]');
          if (control) {
            control.remove();
          }
          node.remove();
        });
        clone.querySelectorAll(".hover-point").forEach(function (node) {
          node.remove();
        });
        clone.querySelectorAll("[data-light]").forEach(function (node) {
          var tagName = node.tagName.toLowerCase();
          if (tagName === "polyline" || tagName === "line") {
            node.setAttribute("stroke", node.dataset.light);
          } else {
            node.setAttribute("fill", node.dataset.light);
          }
        });
        var style = document.createElementNS(SVG_NS, "style");
        style.textContent = PRINT_CSS;
        var background = document.createElementNS(SVG_NS, "rect");
        background.setAttribute("x", "0");
        background.setAttribute("y", "0");
        background.setAttribute("width", width);
        background.setAttribute("height", height);
        background.setAttribute("fill", "#ffffff");
        clone.insertBefore(background, clone.firstChild);
        clone.insertBefore(style, clone.firstChild);
        return {
          markup: new XMLSerializer().serializeToString(clone),
          width: width,
          height: height
        };
      }

      function downloadPng(chart, button) {
        var data = staticSvg(chart);
        var scale = 2;
        var image = new Image();
        image.onload = function () {
          var canvas = document.createElement("canvas");
          canvas.width = data.width * scale;
          canvas.height = data.height * scale;
          var context = canvas.getContext("2d");
          context.fillStyle = "#ffffff";
          context.fillRect(0, 0, canvas.width, canvas.height);
          context.drawImage(image, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(function (blob) {
            var url = URL.createObjectURL(blob);
            var link = document.createElement("a");
            var title = chart.getAttribute("aria-label") || "chart";
            link.href = url;
            link.download = title.toLowerCase().replace(/[^a-z0-9]+/g, "-")
              .replace(/^-|-$/g, "") + ".png";
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            button.disabled = false;
          }, "image/png");
        };
        image.onerror = function () {
          button.disabled = false;
          button.textContent = "failed";
        };
        image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(data.markup);
      }

      document.querySelectorAll(".png-button").forEach(function (button) {
        button.addEventListener("click", function () {
          var figure = button.closest(".usage-figure");
          var chart = figure ? figure.querySelector(".usage-chart") : null;
          if (!chart) {
            return;
          }
          button.disabled = true;
          downloadPng(chart, button);
        });
      });
    })();
"""
