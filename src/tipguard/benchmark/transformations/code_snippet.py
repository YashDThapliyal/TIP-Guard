"""Python-snippet transformation with an AST-only, execution-free decoder.

`encode` writes a short snippet whose *value*, were it run, would be the
hidden text. `decode` never runs it: it parses the source with `ast` and
reconstructs the string by recognising exactly three shapes. Anything else
raises `ValueError("unsupported snippet")`, resource attacks included: the
source length, the reconstruction length and the replace-chain length are
all bounded, and a snippet deep enough to exhaust the walk's recursion is
reported the same way. `encode` respects the same bounds: it falls back to
whichever style fits rather than emitting source its own decoder would
refuse, and raises only if no style fits. This decoder is the seed of the
Phase 4 restricted code analyzer, so it stays deliberately narrow: no
`exec`, no `eval`, no `compile`, no imports, no attribute access on
anything but a string, and no call outside `str.join`, `reversed`, `chr`
and `str.replace`.
"""

import ast
import json
import random

from tipguard.benchmark.transformations.base import Encoded, Family

HINT = "a short Python snippet that builds a string"
UNSUPPORTED = "unsupported snippet"
STYLES = ("chr_codes", "join_reverse", "replace_chain")
MIN_CHUNKS = 3
MAX_CHUNKS = 6
PLACEHOLDERS = ("§", "¤")
MAX_CODE_POINT = 0x10FFFF
# Hostile snippets are cheap to write and expensive to parse or expand, so
# both the source and the reconstruction are bounded before any work starts.
MAX_SOURCE_LENGTH = 20000
MAX_DECODED_LENGTH = 10000
# The encoder emits exactly two chained single-character replacements; a
# longer chain multiplies its input per link, so no other length is accepted.
REPLACE_LINKS = 2
TARGET_NAME = "msg"
PARTS_NAME = "parts"
ALLOWED_FUNCTIONS = frozenset({"chr", "reversed"})
ALLOWED_METHODS = frozenset({"join", "replace"})


def _literal(text: str) -> str:
    """Render `text` as a double-quoted Python string literal."""
    return json.dumps(text, ensure_ascii=False)


def _split(text: str, count: int) -> list[str]:
    """Split `text` into `count` chunks of as-equal-as-possible length."""
    base, extra = divmod(len(text), count)
    sizes = [base + (1 if index < extra else 0) for index in range(count)]
    starts = [sum(sizes[:index]) for index in range(count)]
    return [text[start : start + size] for start, size in zip(starts, sizes, strict=True)]


def _join_reverse_source(text: str, rng: random.Random) -> str:
    """Emit the `parts = [...]` / `"".join(reversed(parts))` pair.

    The chunk count is drawn from `MIN_CHUNKS..MAX_CHUNKS`, then clamped down
    to one chunk per character so that a text shorter than the draw is never
    split into empty chunks; a one- or two-character instruction therefore
    yields one or two chunks rather than three. Real instructions are far
    longer than `MAX_CHUNKS`, so the clamp only ever fires in tests.
    """
    count = min(rng.randint(MIN_CHUNKS, MAX_CHUNKS), max(len(text), 1))
    chunks = _split(text, count)
    listed = ", ".join(_literal(chunk) for chunk in reversed(chunks))
    return f'parts = [{listed}]\nmsg = "".join(reversed(parts))'


def _chr_codes_source(text: str) -> str:
    codes = ", ".join(str(ord(char)) for char in text)
    return f'msg = "".join(chr(c) for c in [{codes}])'


def _replaceable_letters(text: str) -> list[str]:
    """Distinct ASCII letters of `text`, case-sensitive, or none if unusable.

    A text that already contains a placeholder glyph is reported as having
    no replaceable letters, because the placeholders would then no longer
    identify the substituted positions and the round trip would corrupt it.
    """
    if any(placeholder in text for placeholder in PLACEHOLDERS):
        return []
    return sorted({char for char in text if char.isascii() and char.isalpha()})


def _replace_chain_source(text: str, letters: list[str]) -> str:
    first, second = letters
    held = text.replace(first, PLACEHOLDERS[0]).replace(second, PLACEHOLDERS[1])
    return (
        f"msg = {_literal(held)}"
        f".replace({_literal(PLACEHOLDERS[0])}, {_literal(first)})"
        f".replace({_literal(PLACEHOLDERS[1])}, {_literal(second)})"
    )


def _style_source(style: str, text: str, rng: random.Random) -> tuple[str, str]:
    """The snippet source for `style`, with the style actually used.

    `replace_chain` needs two distinct replaceable letters; a text without
    them falls back to `chr_codes`, which is recorded as the style.
    """
    if style == "join_reverse":
        return style, _join_reverse_source(text, rng)
    if style == "replace_chain":
        letters = _replaceable_letters(text)
        if len(letters) >= 2:
            return style, _replace_chain_source(text, rng.sample(letters, 2))
    return "chr_codes", _chr_codes_source(text)


def _first_fitting_source(
    style: str, payload: str, text: str, rng: random.Random
) -> tuple[str, str, int]:
    """The first style whose source fits the cap, and the smallest size seen.

    No style is uniformly the most compact: `chr_codes` costs a few
    characters per code point, while `join_reverse` costs one per character
    plus its escaping, so control characters favour the first and astral
    characters the second. The drawn style is therefore tried first and the
    rest in name order, which keeps the fallback deterministic without
    letting the draw decide whether a text is encodable at all. The drawn
    style is never regenerated, so the common path takes no extra draw.
    """
    smallest = len(payload)
    if smallest <= MAX_SOURCE_LENGTH:
        return style, payload, smallest
    for candidate in sorted(STYLES):
        if candidate == style:
            continue
        used, source = _style_source(candidate, text, rng)
        smallest = min(smallest, len(source))
        if len(source) <= MAX_SOURCE_LENGTH:
            return used, source, len(source)
    return style, payload, smallest


def _require_decodable(text: str, payload: str, smallest: int) -> None:
    """Refuse to emit a payload this module's own decoder would reject.

    Failing here is loud and happens at construction time; the alternative
    is a benchmark case whose prompt cannot be decoded back to its intent.
    """
    if len(text) > MAX_DECODED_LENGTH:
        raise ValueError(
            f"cannot encode {len(text)} characters as a Python snippet: the decoder "
            f"caps a reconstruction at {MAX_DECODED_LENGTH} characters"
        )
    if len(payload) > MAX_SOURCE_LENGTH:
        raise ValueError(
            f"cannot encode {len(text)} characters as a Python snippet: no style fits "
            f"the {MAX_SOURCE_LENGTH}-character source limit, the smallest being "
            f"{smallest} characters"
        )


class CodeSnippetTransformation:
    family = Family.CODE
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        # A long text can produce source past the cap the decoder enforces, so
        # the drawn style may have to give way to one that fits; everything
        # encode returns must survive its own decode.
        style, payload = _style_source(str(rng.choice(sorted(STYLES))), text, rng)
        style, payload, smallest = _first_fitting_source(style, payload, text, rng)
        _require_decodable(text, payload, smallest)
        return Encoded(payload=payload, family=self.family, params={"style": style}, hint=HINT)

    def decode(self, encoded: Encoded) -> str:
        return decode_snippet(encoded.payload)


def decode_snippet(source: str) -> str:
    """Reconstruct the string a supported snippet would build, without running it.

    Every rejection, including one provoked by a resource attack rather than
    by an unrecognised shape, surfaces as `ValueError(UNSUPPORTED)`.
    """
    module = _parse(source)
    try:
        _reject_unsafe_nodes(module)
        for matcher in (_decode_join_reverse, _decode_chr_codes, _decode_replace_chain):
            result = matcher(module)
            if result is not None:
                return _capped(result)
    except (RecursionError, MemoryError) as exc:
        # A deeply nested snippet can exhaust the whitelist walk's mutual
        # recursion; that is an unsupported snippet, not a crash.
        raise ValueError(UNSUPPORTED) from exc
    raise ValueError(UNSUPPORTED)


def _capped(text: str) -> str:
    if len(text) > MAX_DECODED_LENGTH:
        raise ValueError(UNSUPPORTED)
    return text


def _parse(source: str) -> ast.Module:
    if len(source) > MAX_SOURCE_LENGTH:
        raise ValueError(UNSUPPORTED)
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ValueError(UNSUPPORTED) from exc


def _reject_unsafe_nodes(module: ast.Module) -> None:
    """Whitelist walk performed before any shape matching.

    The three matchers below are already exhaustive, so this pass is
    belt-and-braces; it exists so the restriction is stated once, in one
    readable place, rather than being implied by three pattern matches.
    """
    for node in ast.walk(module):
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise ValueError(UNSUPPORTED)
        if isinstance(node, ast.Attribute) and not _is_allowed_attribute(node):
            raise ValueError(UNSUPPORTED)
        if isinstance(node, ast.Call) and not _is_allowed_call(node):
            raise ValueError(UNSUPPORTED)


def _string_constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_string_expression(node: ast.expr) -> bool:
    """True for a string literal or for a call already known to return a string."""
    if _string_constant(node) is not None:
        return True
    return isinstance(node, ast.Call) and _is_allowed_call(node)


def _is_allowed_attribute(node: ast.Attribute) -> bool:
    return node.attr in ALLOWED_METHODS and _is_string_expression(node.value)


def _is_allowed_call(node: ast.Call) -> bool:
    if node.keywords:
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in ALLOWED_FUNCTIONS
    return isinstance(func, ast.Attribute) and _is_allowed_attribute(func)


def _assigned_value(node: ast.stmt, name: str) -> ast.expr | None:
    """The right-hand side of `name = <expr>`, or None for any other statement."""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != name:
        return None
    return node.value


def _join_call(node: ast.expr) -> ast.expr | None:
    """Match `"".join(<argument>)`, returning the argument.

    Both generated styles join on the empty string. A non-empty separator is
    refused: it is a shape the encoder never emits, and it makes
    reconstruction quadratic in the separator length.
    """
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "join":
        return None
    if _string_constant(func.value) != "":
        return None
    return node.args[0]


def _is_single_name_call(node: ast.expr, function: str, argument_name: str) -> bool:
    """Match `function(argument_name)` for a bare name argument."""
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        return False
    func = node.func
    if not isinstance(func, ast.Name) or func.id != function:
        return False
    argument = node.args[0]
    return isinstance(argument, ast.Name) and argument.id == argument_name


def _string_list(node: ast.expr) -> list[str] | None:
    if not isinstance(node, ast.List):
        return None
    chunks: list[str] = []
    for element in node.elts:
        text = _string_constant(element)
        if text is None:
            return None
        chunks.append(text)
    return chunks


def _decode_join_reverse(module: ast.Module) -> str | None:
    if len(module.body) != 2:
        return None
    parts_value = _assigned_value(module.body[0], PARTS_NAME)
    message_value = _assigned_value(module.body[1], TARGET_NAME)
    if parts_value is None or message_value is None:
        return None
    chunks = _string_list(parts_value)
    argument = _join_call(message_value)
    if chunks is None or argument is None:
        return None
    if not _is_single_name_call(argument, "reversed", PARTS_NAME):
        return None
    return "".join(reversed(chunks))


def _code_points(node: ast.expr) -> list[int] | None:
    """Match `chr(c) for c in [<int literals>]`, returning the code points."""
    if not isinstance(node, ast.GeneratorExp) or len(node.generators) != 1:
        return None
    comprehension = node.generators[0]
    target = comprehension.target
    if comprehension.ifs or comprehension.is_async or not isinstance(target, ast.Name):
        return None
    if target.id in ALLOWED_FUNCTIONS:
        # `chr(chr) for chr in [...]` reads as a match but is a TypeError in
        # real Python, so reconstructing it would invent a meaning.
        return None
    if not _is_single_name_call(node.elt, "chr", target.id):
        return None
    if not isinstance(comprehension.iter, ast.List):
        return None
    codes: list[int] = []
    for element in comprehension.iter.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, int):
            return None
        if isinstance(element.value, bool) or not 0 <= element.value <= MAX_CODE_POINT:
            return None
        codes.append(element.value)
    return codes


def _decode_chr_codes(module: ast.Module) -> str | None:
    if len(module.body) != 1:
        return None
    message_value = _assigned_value(module.body[0], TARGET_NAME)
    if message_value is None:
        return None
    argument = _join_call(message_value)
    if argument is None:
        return None
    codes = _code_points(argument)
    if codes is None:
        return None
    return "".join(chr(code) for code in codes)


def _replace_call(node: ast.Call) -> tuple[ast.expr, tuple[str, str]] | None:
    """Match `<receiver>.replace("<c>", "<new>")` for a single-character key."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "replace":
        return None
    if len(node.args) != 2 or node.keywords:
        return None
    old = _string_constant(node.args[0])
    new = _string_constant(node.args[1])
    if old is None or new is None or len(old) != 1:
        return None
    return func.value, (old, new)


def _apply_replacement(text: str, old: str, new: str) -> str:
    """Apply one replacement, refusing one that would expand past the cap."""
    projected = len(text) + text.count(old) * (len(new) - len(old))
    if projected > MAX_DECODED_LENGTH:
        raise ValueError(UNSUPPORTED)
    return text.replace(old, new)


def _decode_replace_chain(module: ast.Module) -> str | None:
    if len(module.body) != 1:
        return None
    node = _assigned_value(module.body[0], TARGET_NAME)
    if node is None:
        return None
    replacements: list[tuple[str, str]] = []
    while isinstance(node, ast.Call):
        matched = _replace_call(node)
        if matched is None or len(replacements) == REPLACE_LINKS:
            return None
        node, replacement = matched
        replacements.append(replacement)
    text = _string_constant(node)
    if text is None or len(replacements) != REPLACE_LINKS:
        return None
    for old, new in reversed(replacements):
        text = _apply_replacement(text, old, new)
    return text
