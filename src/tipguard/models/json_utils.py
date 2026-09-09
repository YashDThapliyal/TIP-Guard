"""Tolerant JSON-object extraction from a raw model reply.

A model asked for a single JSON object routinely wraps it in a markdown
code fence, or adds a sentence of preamble or apology around it despite
being told not to. Every caller that reads a model's JSON reply needs the
same tolerance for that, so it lives here once rather than being
reimplemented per classifier.
"""

import json
import re
from collections.abc import Iterator
from typing import Any

#: A single fenced code block spanning the whole reply, with an optional
#: language tag on the opening fence (` ```json `, ` ``` `, ...). Only a
#: fence that wraps the *entire* text is unwrapped here: a fence embedded
#: partway through prose is left alone, and the object scan below finds the
#: object inside it regardless.
_FULL_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = _FULL_FENCE.match(text.strip())
    return match.group(1) if match else text


#: Size of the first, cheap parse attempt at each candidate `{`. Kept small
#: so a run of failing candidates -- a reply carrying many separate,
#: unmatched braces -- stays cheap: `JSONDecodeError.__init__` computes the
#: line and column of its failure position by scanning from the *start* of
#: whatever text it was given, so handing `raw_decode` the whole remaining
#: text on every failing attempt makes a failure deep into a long reply
#: cost roughly its absolute position, not its local size -- summed over
#: every failing position, that is exactly the O(n^2) blow-up this module
#: used to have, even though each individual parse failed almost instantly.
#: A genuinely large object still parses correctly despite this: a failure
#: that lands exactly at the probe window's edge looks like it ran out of
#: input rather than hit a real syntax error, and `_decode_at` escalates to
#: the full remaining text -- paying its cost -- only in that case, which
#: is rare compared to the many small, immediately-invalid candidates a
#: pathological reply is actually made of.
_PROBE_WINDOW = 4096

#: How close a failure's position has to be to the probe window's own edge
#: to be treated as truncation rather than a genuine syntax error. Not
#: always exactly `len(window)`: a number cut off mid-digit backtracks to
#: report the position right after its last complete digit, one or two
#: characters short of the edge, rather than the edge itself.
_TRUNCATION_SLACK = 20

#: The one failure whose position is *not* near the window's edge even when
#: the failure is pure truncation: an unterminated string reports where the
#: string *started*, which can be anywhere earlier in the window, not where
#: reading it ran out. The message is a stable, undocumented-but-unchanged
#: string from `json.decoder` -- checked in addition to the position, not
#: instead of it, so this only ever widens what counts as "escalate",
#: never narrows a case the position check would have already caught.
_UNTERMINATED_STRING = "Unterminated string starting at"


def _looks_truncated(exc: json.JSONDecodeError, window_len: int) -> bool:
    return exc.pos >= window_len - _TRUNCATION_SLACK or exc.msg == _UNTERMINATED_STRING


def _decode_at(decoder: json.JSONDecoder, text: str, start: int) -> tuple[Any, int]:
    """Parse the JSON value at `text[start]`, returning (value, end).

    Raises `json.JSONDecodeError` or `RecursionError` exactly as
    `decoder.raw_decode(text, start)` would; see `_PROBE_WINDOW` for why
    this tries a bounded slice first instead of calling that directly.
    """
    window = text[start : start + _PROBE_WINDOW]
    try:
        parsed, end = decoder.raw_decode(window, 0)
    except json.JSONDecodeError as exc:
        if not _looks_truncated(exc, len(window)):
            raise
        # `text` (unsliced) is `doc` here, with `start` as the index into
        # it, so `end` is already absolute -- unlike the window branch
        # above, where it is relative to the 0-based slice and has to be
        # shifted by `start` to mean the same thing.
        return decoder.raw_decode(text, start)
    return parsed, start + end


def _brace_matches(text: str) -> dict[int, int]:
    """Maps every `{` index in `text` to the index just past its matching
    `}`, by ordinary LIFO bracket matching; a `{` with no match is absent.

    Iterative and non-recursive, tracking JSON string state so a brace
    inside a string cannot end it early, unlike `json.loads`. Computed once
    per `iter_json_objects` call -- lazily, only if a `RecursionError` ever
    actually occurs -- rather than once per failing candidate: a single
    stack-based pass finds every position's match simultaneously, in the
    same cost as one candidate-sized scan, so later lookups for *other*
    candidates are then O(1) instead of each re-paying a scan of their own.

    LIFO order is what keeps this correct where it matters here: an
    earlier `{` that never closes stays on the stack, unresolved, rather
    than absorbing a `}` that belongs to something opened after it. That
    is what stops a deeply nested, never-closing candidate from swallowing
    a valid, self-contained object that happens to follow it in the text --
    the object's own `{` is pushed after the failed candidate's opens and
    is still the most recent one on the stack when its own `}` arrives, so
    it matches correctly regardless of what never closes underneath it.
    """
    stack: list[int] = []
    matches: dict[int, int] = {}
    in_string = False
    escaped = False
    for index, char in enumerate(text):
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
            stack.append(index)
        elif char == "}" and stack:
            matches[stack.pop()] = index + 1
    return matches


def iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yield every JSON object in `text`, left to right, never raising.

    Tries every `{` in turn with `_decode_at` (a thin wrapper around
    `json.JSONDecoder.raw_decode`, see there), which parses -- or fails --
    starting at an exact offset in a single pass: no separate
    brace-balancing scan, and no rescanning to the end of the text for a
    candidate that never closes. A reply carrying many opening braces that
    never form a valid object is therefore O(n) overall, not O(n^2).

    A successful parse resumes the search right after the object it
    consumed, since everything inside it has already been accounted for --
    this is what lets a caller pull a *second* candidate out of a reply
    that opens with one JSON object and follows it with another, such as a
    model's own reasoning object ahead of its actual answer. An ordinary
    parse failure (invalid JSON) resumes one character past the opening
    brace that failed, rather than skipping the whole span: the failure
    can come from a syntax error partway through a *different*, unrelated
    object than the one enclosing it (a model's own commentary quoting a
    brace pair before its real reply, say), so skipping past the whole
    span could skip a real object hiding inside it.

    A `RecursionError` -- from a candidate too deeply nested for
    `raw_decode` to finish -- is handled differently. The first one to
    occur computes `_brace_matches` once, over the whole text, and every
    later candidate position is then checked against that map *before*
    `_decode_at` is tried at all: a `{` with no recorded match can never
    be the start of a complete JSON object, by ordinary bracket-matching
    logic, regardless of what caused any particular attempt at it to fail,
    so there is nothing to gain by paying for that attempt. Without this,
    resuming one character at a time into the same nested candidate would
    retry the same pathologically deep parse at nearly every one of its
    own nesting levels; this also means a deeply nested but ultimately
    unmatched candidate does not hide an ordinary, self-contained object
    that happens to follow it in the text -- see `_brace_matches` for why
    that holds even though the candidate itself never closes.

    A caller that only wants the first object can use
    `next(iter_json_objects(text), None)` -- `parse_json_object` below does
    exactly that. A caller validating against a schema, such as
    `LLMRiskClassifier`, should instead try each yielded object in turn
    until one validates, since the first *syntactically* valid object in a
    reply is not always the first one a schema will accept.
    """
    stripped = _strip_code_fence(text)
    decoder = json.JSONDecoder()
    matches: dict[int, int] | None = None
    search_from = 0
    while True:
        start = stripped.find("{", search_from)
        if start == -1:
            return
        if matches is not None and start not in matches:
            search_from = start + 1
            continue
        try:
            parsed, end = _decode_at(decoder, stripped, start)
        except RecursionError:
            if matches is None:
                matches = _brace_matches(stripped)
            close = matches.get(start)
            search_from = close if close is not None else start + 1
            continue
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(parsed, dict):
            yield parsed
        search_from = max(end, start + 1)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in `text`, or None. See `iter_json_objects`."""
    return next(iter_json_objects(text), None)
