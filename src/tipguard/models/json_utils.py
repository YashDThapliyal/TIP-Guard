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


def _balanced_object_at(text: str, start: int) -> str | None:
    """The balanced `{...}` beginning exactly at `text[start]`, or None.

    Braces inside a JSON string are not counted: without tracking string
    state, a rationale that itself contains `}` (an attacker-supplied
    prompt reflected back by the model, say) would close the object early.
    """
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
    """Extract and parse the first *parseable* JSON object in `text`.

    Tries every `{` in the text in turn, left to right, rather than
    committing to the first one: a `{` that opens a balanced but invalid
    span -- unquoted keys, or a `{` a model used in prose before the real
    object ("I will use the shape {risk, categories} -- here it is:
    {...}") -- is skipped in favour of the next one instead of failing the
    whole reply. Returns None -- never raises -- when no `{` in the text
    ever yields a JSON object.

    A candidate whose parse overflows the interpreter's recursion limit
    aborts the search outright rather than being skipped like an ordinary
    parse failure: a `RecursionError` means the input is pathologically
    nested, and every later, shallower starting point is a substring of
    that same nesting, so retrying them cannot turn up a different object
    and would only re-pay a large chunk of the failed parse's cost for
    every remaining `{` in the text.
    """
    stripped = _strip_code_fence(text)
    search_from = 0
    while True:
        start = stripped.find("{", search_from)
        if start == -1:
            return None
        candidate = _balanced_object_at(stripped, start)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except RecursionError:
                return None
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        search_from = start + 1
