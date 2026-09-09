"""Tolerant JSON-object extraction from a raw model reply.

A model asked for a single JSON object routinely wraps it in a markdown
code fence, or adds a sentence of preamble or apology around it despite
being told not to. Every caller that reads a model's JSON reply needs the
same tolerance for that, so it lives here once rather than being
reimplemented per classifier.
"""

import json
import re
from typing import Any

#: A single fenced code block spanning the whole reply, with an optional
#: language tag on the opening fence (` ```json `, ` ``` `, ...). Only a
#: fence that wraps the *entire* text is unwrapped here: a fence embedded
#: partway through prose is left alone, and the balanced-object scan below
#: finds the object inside it regardless.
_FULL_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = _FULL_FENCE.match(text.strip())
    return match.group(1) if match else text


def _first_balanced_object(text: str) -> str | None:
    """The first `{...}` in `text` whose braces balance, or None.

    Braces inside a JSON string are not counted: without tracking string
    state, a rationale that itself contains `}` (an attacker-supplied
    prompt reflected back by the model, say) would close the object early.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract and parse the first balanced JSON object in `text`.

    Returns None -- never raises -- for text with no object, an unbalanced
    or otherwise malformed object, or a balanced value that parses to
    something other than a JSON object (a bare array or scalar). Callers
    that need to keep running when a model's reply is unusable rely on this
    never raising.
    """
    candidate = _first_balanced_object(_strip_code_fence(text))
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
