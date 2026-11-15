"""Pygments styles in Catppuccin (https://catppuccin.com/palette), for the API reference's code: Latte in light mode,
Macchiato in dark, as the rest of the docs. Named in conf.py by their import path (`pygments_style`,
`pygments_dark_style`), so no package is needed."""

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    Token,
)

LATTE = {
    "rosewater": "#dc8a78", "pink": "#ea76cb", "mauve": "#8839ef", "red": "#d20f39", "maroon": "#e64553",
    "peach": "#fe640b", "yellow": "#df8e1d", "green": "#40a02b", "teal": "#179299", "sky": "#04a5e5",
    "blue": "#1e66f5", "text": "#4c4f69", "subtext0": "#6c6f85", "overlay2": "#7c7f93", "surface0": "#ccd0da",
    "mantle": "#e6e9ef",
}
MACCHIATO = {
    "rosewater": "#f4dbd6", "pink": "#f5bde6", "mauve": "#c6a0f6", "red": "#ed8796", "maroon": "#ee99a0",
    "peach": "#f5a97f", "yellow": "#eed49f", "green": "#a6da95", "teal": "#8bd5ca", "sky": "#91d7e3",
    "blue": "#8aadf4", "text": "#cad3f5", "subtext0": "#a5adcb", "overlay2": "#939ab7", "surface0": "#363a4f",
    "mantle": "#1e2030",
}


def _styles(c):
    """Catppuccin's syntax colours: keywords mauve, strings green, numbers peach, functions blue, types yellow, ..."""
    return {
        Token: c["text"],
        Text: c["text"],
        Error: c["red"],
        Comment: f"italic {c['overlay2']}",
        Comment.Preproc: c["pink"],
        Keyword: c["mauve"],
        Keyword.Constant: c["peach"],
        Keyword.Type: c["yellow"],
        Name: c["text"],
        Name.Builtin: c["red"],
        Name.Builtin.Pseudo: c["red"],
        Name.Function: c["blue"],
        Name.Function.Magic: c["blue"],
        Name.Class: c["yellow"],
        Name.Decorator: c["peach"],
        Name.Exception: c["maroon"],
        Name.Namespace: c["yellow"],
        Name.Attribute: c["yellow"],
        Name.Tag: c["blue"],
        Name.Variable: c["text"],
        Name.Variable.Magic: c["red"],
        String: c["green"],
        String.Doc: c["green"],
        String.Escape: c["pink"],
        String.Interpol: c["pink"],
        String.Regex: c["pink"],
        Number: c["peach"],
        Operator: c["sky"],
        Operator.Word: c["mauve"],
        Punctuation: c["overlay2"],
        Generic.Deleted: c["red"],
        Generic.Inserted: c["green"],
        Generic.Heading: f"bold {c['blue']}",
        Generic.Subheading: f"bold {c['mauve']}",
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Generic.Prompt: c["overlay2"],
        Generic.Output: c["subtext0"],
        Generic.Error: c["red"],
    }


class LatteStyle(Style):
    """Catppuccin Latte."""

    name = "catppuccin-latte"
    background_color = LATTE["mantle"]
    highlight_color = LATTE["surface0"]
    styles = _styles(LATTE)


class MacchiatoStyle(Style):
    """Catppuccin Macchiato."""

    name = "catppuccin-macchiato"
    background_color = MACCHIATO["mantle"]
    highlight_color = MACCHIATO["surface0"]
    styles = _styles(MACCHIATO)
