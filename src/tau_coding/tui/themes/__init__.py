# [PACKAGE DOC]
# Package defining JSON-based TUI themes for the Textual frontend. Themes are
# data: built-in themes (tau-dark, tau-light, high-contrast) ship as JSON
# files in this package and are parsed at import time by the same strict
# parser used for user themes (from <Tau home>/themes and <project>/.tau/themes).
# Provides the theme dataclasses, the transcript role list, a parser that
# validates fields, vars substitution, colors (Rich and Textual), role
# styles, darkness inference and syntax themes; a process-global registry of
# custom themes; lookup; conversion to Textual themes / CSS variables; and a
# directory loader that reports diagnostics instead of failing.
# Side effects at import: reads the bundled JSON theme files.
"""JSON-defined TUI themes for Tau.

Themes are data, not code. The built-in themes ship as JSON files next to this
module and load through the same parser as user themes, which live in
``<Tau home>/themes/*.json`` and ``<project>/.tau/themes/*.json``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cache
from importlib.resources import files
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Literal, get_args

from rich.color import Color, ColorParseError
from rich.errors import StyleSyntaxError
from rich.style import Style
from textual.color import Color as TextualColor
from textual.color import ColorParseError as TextualColorParseError
from textual.theme import Theme

from tau_coding.resources import ResourceDiagnostic


class TuiThemeError(ValueError):
    # [CLASS DOC]
    # Exception (ValueError) raised when a theme definition is invalid; the
    # message aggregates all problems found.
    """Raised when a TUI theme definition is invalid."""


@dataclass(frozen=True, slots=True)
class TuiRoleStyle:
    # [CLASS DOC]
    # Frozen dataclass of colors for one transcript role block: a border color
    # and a body Rich style string.
    """Colors for one transcript role block."""

    border: str
    body: str


@dataclass(frozen=True, slots=True)
class TuiTheme:
    # [CLASS DOC]
    # Frozen dataclass of a fully resolved theme: name, dark flag, a set of
    # single-color fields for screen, chrome, sidebar, borders, prompt,
    # autocomplete, accent/success/error, tool text, highlights and markdown
    # elements, three completion Rich style fields, a Pygments syntax theme name,
    # and a mapping from transcript role to role style.
    """Resolved visual theme for Tau's built-in Textual frontend."""

    name: str
    dark: bool
    screen_background: str
    screen_text: str
    chrome_background: str
    chrome_text: str
    muted_text: str
    sidebar_background: str
    border: str
    transcript_background: str
    prompt_background: str
    prompt_text: str
    prompt_border: str
    autocomplete_background: str
    accent: str
    success: str
    error: str
    tool_success_text: str
    tool_error_text: str
    highlight_background: str
    highlight_text: str
    markdown_heading: str
    markdown_table_header: str
    markdown_table_border: str
    markdown_inline_code: str
    markdown_code_block_background: str
    markdown_link: str
    markdown_bullet: str
    completion_selected: str
    completion_selected_description: str
    completion_description: str
    syntax_theme: str
    role_styles: dict[str, TuiRoleStyle]


type TuiThemeName = str

THEME_COLOR_FIELDS: tuple[str, ...] = tuple(
    theme_field.name
    for theme_field in fields(TuiTheme)
    if theme_field.name not in {"name", "dark", "syntax_theme", "role_styles"}
)

# Single source of truth for transcript roles: the theme schema requires a
# style for each, and tui.state aliases this as ChatItemRole.
TranscriptRole = Literal[
    "user",
    "assistant",
    "tool",
    "error",
    "status",
    "thinking",
    "skill",
    "custom",
    "branch_summary",
    "compaction_summary",
]
TRANSCRIPT_ROLES: tuple[str, ...] = get_args(TranscriptRole)

_TOP_LEVEL_FIELDS = {"$schema", "name", "dark", "vars", "syntax_theme", "colors", "roles"}

# Fields that feed Textual CSS variables and must be a single color, unlike
# the completion fields and role bodies, which are full Rich style strings.
_RICH_STYLE_FIELDS = {
    "completion_selected",
    "completion_selected_description",
    "completion_description",
}

# Single-color fields only ever rendered through Rich. Every other color field
# reaches Textual too (CSS variables, Theme slots, or widget styles), and Rich
# accepts colors Textual rejects (bright_red, grey50, color(1), default), so
# those fields must parse under both libraries. New fields default to the
# strict dual check: a field missing from this set rejects a theme with a
# diagnostic instead of crashing the TUI when the theme is applied.
_RICH_ONLY_COLOR_FIELDS = {
    "tool_success_text",
    "tool_error_text",
}

# Var names that would corrupt Rich style strings during token substitution.
_RICH_STYLE_KEYWORDS = frozenset(
    {
        "on",
        "not",
        "none",
        "default",
        "b",
        "bold",
        "d",
        "dim",
        "i",
        "italic",
        "u",
        "underline",
        "uu",
        "underline2",
        "s",
        "strike",
        "r",
        "reverse",
        "blink",
        "blink2",
        "conceal",
        "o",
        "overline",
        "frame",
        "encircle",
        "link",
    }
)


def parse_tui_theme_json(data: object) -> TuiTheme:
    # [FUNCTION DOC]
    # Parse and validate a theme from JSON data, collecting all problems before
    # failing. Requires an object; flags unknown top-level fields; parses the
    # name (non-empty, no slash), vars (named single colors usable as tokens),
    # colors (every theme color field, with vars substituted, validated per field
    # kind), roles (every transcript role with border and body), dark (explicit
    # bool or inferred from screen background luminance), and syntax theme
    # (known Pygments style or ansi default based on darkness).
    # Output: TuiTheme. Raises TuiThemeError listing every problem (prefixed by
    # the theme name when known). No side effects beyond a cached Pygments style
    # lookup.
    """Parse a theme from JSON-compatible data, reporting all problems at once."""
    if not isinstance(data, dict):
        raise TuiThemeError("Theme must be a JSON object")

    problems: list[str] = []
    for key in sorted(set(data) - _TOP_LEVEL_FIELDS):
        problems.append(f"unknown field: {key}")

    name = _parse_name(data.get("name"), problems)
    variables = _parse_vars(data.get("vars", {}), problems)
    colors = _parse_colors(data.get("colors"), variables, problems)
    role_styles = _parse_roles(data.get("roles"), variables, problems)
    dark = _parse_dark(data.get("dark"), colors, problems)
    syntax_theme = _parse_syntax_theme(data.get("syntax_theme"), dark=dark, problems=problems)

    if problems:
        label = f"theme {name!r}" if name else "theme"
        raise TuiThemeError(f"Invalid {label}: " + "; ".join(problems))
    return TuiTheme(
        name=name,
        dark=dark,
        syntax_theme=syntax_theme,
        role_styles=role_styles,
        **colors,
    )


def _parse_name(value: object, problems: list[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: validate the theme name is a non-empty string without "/",
    # appending problems; returns the stripped name or empty string.
    if not isinstance(value, str) or not value.strip():
        problems.append("name must be a non-empty string")
        return ""
    name = value.strip()
    if "/" in name:
        problems.append("name must not contain '/'")
    return name


def _parse_vars(value: object, problems: list[str]) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: validate the vars object. Each name must be a non-empty string not
    # equal (case-insensitively) to a Rich style keyword; each value must be a
    # single whitespace-free token that Rich parses as a color. Invalid entries
    # are reported and skipped. Returns mapping name -> color.
    if not isinstance(value, dict):
        problems.append("vars must be an object")
        return {}
    variables: dict[str, str] = {}
    for var_name, var_value in value.items():
        name_allowed = isinstance(var_name, str) and var_name
        if not name_allowed or var_name.lower() in _RICH_STYLE_KEYWORDS:
            problems.append(f"vars name is not allowed: {var_name!r}")
            continue
        if (
            not isinstance(var_value, str)
            or len(var_value.split()) != 1
            or _color_problem(var_value) is not None
        ):
            problems.append(f"vars value must be a single color: {var_name}")
            continue
        variables[var_name] = var_value
    return variables


def _substitute_vars(value: str, variables: dict[str, str]) -> str:
    # [FUNCTION DOC]
    # HELPER: replace whitespace-separated tokens that match a var name with its
    # color; tokens are re-joined with single spaces. Returns the input unchanged
    # when there are no vars.
    if not variables:
        return value
    return " ".join(variables.get(token, token) for token in value.split())


def _parse_colors(
    value: object,
    variables: dict[str, str],
    problems: list[str],
) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: validate the colors object: reports missing and unknown fields,
    # requires non-empty strings, substitutes vars, then validates completion
    # fields as Rich styles, Rich-only fields as Rich colors, and all others as
    # colors acceptable to both Rich and Textual. Returns mapping of valid
    # resolved fields; problems are appended to the list.
    if not isinstance(value, dict):
        problems.append("colors must be an object")
        return {}
    missing = [field_name for field_name in THEME_COLOR_FIELDS if field_name not in value]
    if missing:
        problems.append("colors missing: " + ", ".join(missing))
    unknown = sorted(set(value) - set(THEME_COLOR_FIELDS))
    if unknown:
        problems.append("colors unknown: " + ", ".join(unknown))
    colors: dict[str, str] = {}
    for field_name in THEME_COLOR_FIELDS:
        if field_name not in value:
            continue
        raw = value[field_name]
        if not isinstance(raw, str) or not raw.strip():
            problems.append(f"colors.{field_name} must be a non-empty string")
            continue
        resolved = _substitute_vars(raw.strip(), variables)
        if field_name in _RICH_STYLE_FIELDS:
            error = _style_problem(resolved)
        elif field_name in _RICH_ONLY_COLOR_FIELDS:
            error = _color_problem(resolved)
        else:
            error = _color_problem(resolved) or _textual_color_problem(resolved)
        if error is not None:
            problems.append(f"colors.{field_name} {error}")
            continue
        colors[field_name] = resolved
    return colors


def _parse_roles(
    value: object,
    variables: dict[str, str],
    problems: list[str],
) -> dict[str, TuiRoleStyle]:
    # [FUNCTION DOC]
    # HELPER: validate the roles object: reports missing/unknown roles; each role
    # must be an object with exactly border and body strings. Border must parse
    # as a color in Rich and Textual; body must be a valid Rich style whose
    # foreground/background colors Textual also accepts. Returns mapping role ->
    # role style for fully valid roles; problems appended.
    if not isinstance(value, dict):
        problems.append("roles must be an object")
        return {}
    missing = [role for role in TRANSCRIPT_ROLES if role not in value]
    if missing:
        problems.append("roles missing: " + ", ".join(missing))
    unknown = sorted(set(value) - set(TRANSCRIPT_ROLES))
    if unknown:
        problems.append("roles unknown: " + ", ".join(unknown))
    role_styles: dict[str, TuiRoleStyle] = {}
    for role in TRANSCRIPT_ROLES:
        if role not in value:
            continue
        raw = value[role]
        if not isinstance(raw, dict) or set(raw) != {"border", "body"}:
            problems.append(f"roles.{role} must be an object with 'border' and 'body'")
            continue
        border, body = raw["border"], raw["body"]
        if not isinstance(border, str) or not isinstance(body, str):
            problems.append(f"roles.{role} border and body must be strings")
            continue
        resolved_border = _substitute_vars(border.strip(), variables)
        resolved_body = _substitute_vars(body.strip(), variables)
        # Borders feed Textual's styles.border_left as well as Rich tables.
        border_error = _color_problem(resolved_border) or _textual_color_problem(resolved_border)
        if border_error is not None:
            problems.append(f"roles.{role}.border {border_error}")
        # Body colors also feed Textual's styles.color/background in the
        # transcript, so both style components must satisfy Textual too.
        body_error = _style_problem(resolved_body) or _textual_style_colors_problem(resolved_body)
        if body_error is not None:
            problems.append(f"roles.{role}.body {body_error}")
        if border_error is None and body_error is None:
            role_styles[role] = TuiRoleStyle(border=resolved_border, body=resolved_body)
    return role_styles


def _color_problem(value: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: return an error phrase if Rich cannot parse the value as a color,
    # else None.
    try:
        Color.parse(value)
    except ColorParseError:
        return f"is not a valid color: {value!r}"
    return None


def _textual_color_problem(value: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: return an error phrase if Textual cannot parse the value as a
    # color, else None.
    try:
        TextualColor.parse(value)
    except TextualColorParseError:
        return f"is not a color Textual accepts: {value!r}"
    return None


def _textual_style_colors_problem(value: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: parse a Rich style (caller must have validated it already, since
    # parse errors are not caught here) and check its named foreground and
    # background colors with Textual; returns the first error phrase or None.
    style = Style.parse(value)
    for color in (style.color, style.bgcolor):
        if color is not None and color.name is not None:
            error = _textual_color_problem(color.name)
            if error is not None:
                return error
    return None


def _style_problem(value: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: return an error phrase if the value is not a valid Rich style
    # (syntax or color error), else None.
    try:
        Style.parse(value)
    except (StyleSyntaxError, ColorParseError):
        return f"is not a valid style: {value!r}"
    return None


def _parse_dark(value: object, colors: dict[str, str], problems: list[str]) -> bool:
    # [FUNCTION DOC]
    # HELPER: return an explicit boolean dark flag; a non-null non-bool value is
    # reported as a problem; otherwise infer darkness from the screen background.
    if isinstance(value, bool):
        return value
    if value is not None:
        problems.append("dark must be a boolean")
    return _is_dark_background(colors.get("screen_background", ""))


def _is_dark_background(value: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: infer darkness from the first token of a color string using
    # relative luminance (Rec. 709 weights) below 0.5; empty or unparseable
    # values are treated as dark.
    tokens = value.split()
    if not tokens:
        return True
    try:
        color = Color.parse(tokens[0])
    except ColorParseError:
        return True
    red, green, blue = color.get_truecolor()
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return luminance < 0.5


def _parse_syntax_theme(value: object, *, dark: bool, problems: list[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: return the syntax theme: ansi_dark/ansi_light by darkness when
    # unset; the value when it is a known Pygments style or ansi theme; otherwise
    # records a problem and returns ansi_dark.
    if value is None:
        return "ansi_dark" if dark else "ansi_light"
    if not isinstance(value, str) or value not in _known_syntax_themes():
        problems.append(f"unknown syntax_theme: {value!r}")
        return "ansi_dark"
    return value


@cache
def _known_syntax_themes() -> frozenset[str]:
    # [FUNCTION DOC]
    # HELPER: cached set of all Pygments style names plus ansi_dark/ansi_light
    # (imports Pygments lazily).
    from pygments.styles import get_all_styles

    return frozenset(get_all_styles()) | {"ansi_dark", "ansi_light"}


def _load_builtin_theme(filename: str) -> TuiTheme:
    # [FUNCTION DOC]
    # HELPER: read a bundled theme JSON file from this package's resources and
    # parse it. Raises on read/parse/validation failure (at import time).
    payload = (files(__package__) / filename).read_text(encoding="utf-8")
    return parse_tui_theme_json(loads(payload))


TAU_DARK_THEME = _load_builtin_theme("tau-dark.json")
TAU_LIGHT_THEME = _load_builtin_theme("tau-light.json")
HIGH_CONTRAST_THEME = _load_builtin_theme("high-contrast.json")

_BUILTIN_THEMES: dict[str, TuiTheme] = {
    theme.name: theme for theme in (TAU_DARK_THEME, TAU_LIGHT_THEME, HIGH_CONTRAST_THEME)
}
BUILTIN_TUI_THEME_NAMES: tuple[TuiThemeName, ...] = tuple(_BUILTIN_THEMES)

_custom_themes: dict[str, TuiTheme] = {}


def set_custom_tui_themes(themes: Mapping[str, TuiTheme]) -> None:
    # [FUNCTION DOC]
    # Replace the process-global custom theme registry with the given mapping
    # of name -> theme. Side effect: mutates module state.
    """Replace the registered custom themes."""
    _custom_themes.clear()
    _custom_themes.update(themes)


def available_tui_theme_names() -> tuple[TuiThemeName, ...]:
    # [FUNCTION DOC]
    # Return built-in theme names (in definition order) followed by custom theme
    # names sorted alphabetically.
    """Return built-in theme names followed by sorted custom theme names."""
    return (*BUILTIN_TUI_THEME_NAMES, *sorted(_custom_themes))


def get_tui_theme(name: TuiThemeName = "tau-dark") -> TuiTheme:
    # [FUNCTION DOC]
    # Return a theme by name (default tau-dark), checking built-ins first and
    # then the custom registry. Raises KeyError when unknown.
    """Return a built-in or registered custom theme by name."""
    if name in _BUILTIN_THEMES:
        return _BUILTIN_THEMES[name]
    return _custom_themes[name]


def textual_theme_for_tui_theme(theme_name: TuiThemeName) -> Theme:
    # [FUNCTION DOC]
    # Look up a Tau theme by name and build a Textual Theme, mapping accent to
    # primary/accent, prompt border to secondary, markdown bullet to warning,
    # error/success, screen text/background to foreground/background, chrome to
    # surface, sidebar to panel, the dark flag, and the Tau CSS variables.
    # Raises KeyError for unknown names.
    """Map a Tau theme to Textual's native theme type."""
    theme = get_tui_theme(theme_name)
    return Theme(
        name=theme.name,
        primary=theme.accent,
        secondary=theme.prompt_border,
        warning=theme.markdown_bullet,
        error=theme.error,
        success=theme.success,
        accent=theme.accent,
        foreground=theme.screen_text,
        background=theme.screen_background,
        surface=theme.chrome_background,
        panel=theme.sidebar_background,
        dark=theme.dark,
        variables=theme_css_variables(theme),
    )


def theme_css_variables(theme: TuiTheme) -> dict[str, str]:
    # [FUNCTION DOC]
    # Return the mapping of Textual CSS variable names (tau-* and footer-*) to
    # color values from a resolved theme; the tool-running color comes from the
    # tool role border. Raises KeyError if the tool role is missing.
    """Return Textual CSS variables for a resolved Tau theme."""
    return {
        "tau-screen-background": theme.screen_background,
        "tau-screen-text": theme.screen_text,
        "tau-chrome-background": theme.chrome_background,
        "tau-chrome-text": theme.chrome_text,
        "tau-muted-text": theme.muted_text,
        "tau-sidebar-background": theme.sidebar_background,
        "tau-border": theme.border,
        "tau-transcript-background": theme.transcript_background,
        "tau-prompt-background": theme.prompt_background,
        "tau-prompt-text": theme.prompt_text,
        "tau-prompt-border": theme.prompt_border,
        "tau-autocomplete-background": theme.autocomplete_background,
        "tau-accent": theme.accent,
        "tau-tool-running": theme.role_styles["tool"].border,
        "tau-highlight-background": theme.highlight_background,
        "tau-highlight-text": theme.highlight_text,
        "tau-markdown-highlight": theme.markdown_heading,
        "tau-markdown-table-header": theme.markdown_table_header,
        "tau-markdown-table-border": theme.markdown_table_border,
        "tau-markdown-inline-code": theme.markdown_inline_code,
        "tau-markdown-code-block-background": theme.markdown_code_block_background,
        "tau-markdown-link": theme.markdown_link,
        "tau-markdown-bullet": theme.markdown_bullet,
        "footer-background": theme.chrome_background,
        "footer-foreground": theme.chrome_text,
        "footer-description-background": theme.chrome_background,
        "footer-description-foreground": theme.chrome_text,
        "footer-key-background": theme.chrome_background,
        "footer-key-foreground": theme.accent,
        "footer-item-background": theme.chrome_background,
    }


def load_custom_tui_themes(
    theme_dirs: Sequence[Path],
) -> tuple[dict[str, TuiTheme], list[ResourceDiagnostic]]:
    # [FUNCTION DOC]
    # Load user themes from directories given in increasing precedence order.
    # Iterates directories from highest precedence, skipping non-directories,
    # and their *.json files in sorted order. Unreadable or invalid JSON files
    # and invalid themes produce invalid-theme diagnostics; themes whose names
    # shadow a built-in, or duplicate an already-loaded (higher-precedence or
    # earlier) theme, produce collision diagnostics and are skipped.
    # Output: tuple of (mapping name -> theme, list of diagnostics). Side
    # effects: filesystem reads only; does not register themes.
    """Load custom themes from directories given in increasing precedence order.

    Invalid files are skipped with a diagnostic; on duplicate names the theme
    from the higher-precedence directory (or earlier file within one directory)
    wins and the loser is reported as a collision.
    """
    themes: dict[str, TuiTheme] = {}
    diagnostics: list[ResourceDiagnostic] = []
    for directory in reversed(list(theme_dirs)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = loads(path.read_text(encoding="utf-8"))
            except (OSError, JSONDecodeError, UnicodeDecodeError) as error:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="invalid-theme",
                        message=f"could not parse theme JSON: {error}",
                        path=path,
                    )
                )
                continue
            try:
                theme = parse_tui_theme_json(data)
            except TuiThemeError as error:
                diagnostics.append(
                    ResourceDiagnostic(kind="invalid-theme", message=str(error), path=path)
                )
                continue
            if theme.name in _BUILTIN_THEMES:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="collision",
                        message=f"theme {theme.name!r} shadows a built-in theme and was ignored",
                        path=path,
                        name=theme.name,
                    )
                )
                continue
            if theme.name in themes:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="collision",
                        message=f"theme {theme.name!r} is already defined with higher precedence",
                        path=path,
                        name=theme.name,
                    )
                )
                continue
            themes[theme.name] = theme
    return themes, diagnostics
