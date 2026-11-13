"""Turn the Markdown in the bindings' docstrings into reStructuredText, for Sphinx.

The docstrings are written for Python's `help()` and IDEs, in Markdown: fenced code blocks and `single backtick`
code. reStructuredText reads single backticks as italic "interpreted text" and ignores fences, so convert both before
napoleon and autoapi see them. (Google-style sections, "Args:" and so on, are left to napoleon.)
"""

import re

FENCE = re.compile(r"^(\s*)```(\w*)\s*$")
INLINE_CODE = re.compile(r"(?<![`\w])`([^`\n]+)`(?![`\w])")


def convert(lines):
    out = []
    in_fence = False
    indent = ""
    for line in lines:
        fence = FENCE.match(line)
        if fence and not in_fence:
            indent = fence.group(1)
            language = fence.group(2) or "text"
            out += [f"{indent}.. code-block:: {language}", ""]
            in_fence = True
        elif fence and in_fence:
            out.append("")
            in_fence = False
        elif in_fence:
            out.append(f"{indent}   {line[len(indent):]}" if line.strip() else "")
        else:
            out.append(INLINE_CODE.sub(r"``\1``", line))
    return out


def process_docstring(app, what, name, obj, options, lines):
    lines[:] = convert(lines)


def setup(app):
    # Before napoleon (priority 500), so that its sections see the converted text
    app.connect("autodoc-process-docstring", process_docstring, priority=400)
    return {"parallel_read_safe": True}
