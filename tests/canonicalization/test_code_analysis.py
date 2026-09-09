"""The restricted code analyzer: what it computes, and what it refuses."""

import json
from pathlib import Path

import pytest

from tipguard.benchmark.io import load_cases
from tipguard.canonicalization.code_analysis import (
    BUDGET_EXCEEDED,
    MAX_SOURCE_LENGTH,
    RestrictedCodeAnalyzer,
)
from tipguard.canonicalization.detector import _extract_code


def _analyze(source: str) -> object:
    return RestrictedCodeAnalyzer().analyze(source)


#: The three shapes the benchmark generator emits, which a defence has to
#: handle without being told which one it is looking at.
GENERATOR_STYLES = (
    ('msg = "".join(chr(c) for c in [104, 105])', "hi"),
    ('msg = "he§§o".replace("§", "l")', "hello"),
    ('parts = ["lo", "hel"]\nmsg = "".join(reversed(parts))', "hello"),
)


@pytest.mark.parametrize(("source", "expected"), GENERATOR_STYLES)
def test_the_generator_styles_resolve(source: str, expected: str) -> None:
    analysis = _analyze(source)
    assert analysis.rejected_reason is None
    assert analysis.result == expected


#: Everything a prompt might carry that this must refuse. These are the
#: reason the module exists: an attacker who can put code in a prompt can put
#: anything in one, so the whitelist has to be closed rather than filtered.
REJECTED = (
    "import os",
    "from os import system",
    'open("x")',
    '__import__("os").system("id")',
    'eval("1+1")',
    'exec("x=1")',
    'compile("1", "s", "eval")',
    "os.system('id')",
    "os.path.join('a', 'b')",
    "while True:\n    pass",
    "for i in range(10):\n    pass",
    "def f():\n    return 1",
    "class C:\n    pass",
    "lambda: 1",
    "x = [i for i in range(10**6)]",
    "globals()",
    "locals()",
    "().__class__.__bases__",
    'getattr(str, "join")',
    "x = 1 if True else 2",
    "async def f():\n    pass",
    "with open('x') as f:\n    pass",
    "assert False",
    "del x",
    "raise ValueError",
    "x += 1",
    "x = yield 1",
)


@pytest.mark.parametrize("source", REJECTED)
def test_anything_outside_the_subset_is_rejected_with_a_reason(source: str) -> None:
    analysis = _analyze(source)
    assert analysis.result is None
    assert analysis.rejected_reason


def test_multiplication_is_not_whitelisted() -> None:
    # The brief asks for this explicitly. `"a" * 10**7` is one node that
    # allocates megabytes, so it is refused for being outside the subset
    # rather than caught afterwards by the length cap.
    analysis = _analyze('msg = "a" * 10000000')
    assert analysis.result is None
    assert analysis.rejected_reason == "operator Mult"


def test_a_long_comprehension_is_refused_before_it_runs() -> None:
    analysis = _analyze("msg = [c for c in " + str(list(range(20_000))) + "]")
    assert analysis.rejected_reason == BUDGET_EXCEEDED


def test_a_step_budget_stops_a_deep_expression() -> None:
    analysis = _analyze('msg = "a"' + ' + "a"' * 20_000)
    assert analysis.result is None
    assert analysis.rejected_reason == BUDGET_EXCEEDED


def test_a_long_source_is_refused_before_parsing() -> None:
    # The bound comes before `ast.parse`, because parsing is itself
    # attacker-controlled work.
    analysis = _analyze("x = 1\n" * MAX_SOURCE_LENGTH)
    assert analysis.rejected_reason == BUDGET_EXCEEDED


def test_string_growth_is_capped_as_it_happens() -> None:
    # Not caught only at the end: each concatenation is checked, so the
    # snippet cannot allocate the whole value first.
    source = 'a = "x" * 1\n'  # rejected earlier, so build growth with join
    del source
    parts = '["' + '","'.join(["y" * 900] * 20) + '"]'
    analysis = _analyze(f'msg = "".join({parts})')
    assert analysis.rejected_reason == BUDGET_EXCEEDED


def test_an_empty_replace_pattern_is_refused() -> None:
    # `"abc".replace("", "x")` inserts between every character, a cheap way
    # to multiply a string.
    assert _analyze('msg = "abc".replace("", "x")').rejected_reason


def test_an_unknown_name_is_never_resolved_against_builtins() -> None:
    # The failure has to be "unknown name", not a successful lookup of the
    # real builtin, which is what keeps the whitelist closed.
    analysis = _analyze("msg = print")
    assert analysis.rejected_reason == "unknown name 'print'"


def test_a_module_attribute_chain_is_rejected() -> None:
    # `get` is not a whitelisted method, so the chain is refused before the
    # receiver is even evaluated. Either failure is correct; what matters is
    # that no part of it resolves to the real `os`.
    analysis = _analyze('msg = os.environ.get("HOME")')
    assert analysis.result is None
    assert analysis.rejected_reason in {"method 'get'", "unknown name 'os'"}


def test_the_operations_used_are_reported() -> None:
    # A trace can say what a snippet did without quoting the snippet, which
    # matters because a decoded payload may carry a protected value.
    analysis = _analyze('msg = "he§§o".replace("§", "l").upper()')
    assert analysis.result == "HELLO"
    assert analysis.operations == ("replace", "upper")


def test_a_loop_variable_does_not_leak_out_of_its_comprehension() -> None:
    analysis = _analyze('parts = [chr(c) for c in [104, 105]]\nmsg = "".join(parts)')
    assert analysis.result == "hi"
    assert _analyze("parts = [chr(c) for c in [104]]\nmsg = c").rejected_reason


def test_a_snippet_that_builds_no_string_is_rejected() -> None:
    assert _analyze("x = 1").rejected_reason == "snippet does not build a string"


def test_the_module_contains_no_execution_primitive() -> None:
    """The guarantee the whole module rests on, asserted rather than assumed.

    A reviewer should not have to take the docstring's word for it, and a
    later edit that reaches for `eval` to handle one more shape should fail
    here rather than in review.
    """
    import ast

    tree = ast.parse(
        Path("src/tipguard/canonicalization/code_analysis.py").read_text(encoding="utf-8")
    )
    # Parsed rather than grepped: the module names these primitives in its own
    # prose to explain why it does not use them, and a text scan cannot tell
    # a docstring from a call. A call node cannot hide in a comment.
    banned = {"eval", "exec", "compile", "__import__", "globals", "locals", "getattr"}
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called & banned), f"module calls {sorted(called & banned)}"


def test_every_code_case_in_the_corpus_resolves() -> None:
    """The accuracy side: the analyzer has to actually read the corpus.

    A whitelist narrow enough to be safe and too narrow to be useful would
    pass every rejection test above and still be worthless, so this measures
    what it recovers.
    """
    cases = [
        case
        for case in load_cases(Path("data/generated/tipguard-v1.jsonl"))
        if case.transformation == "code"
    ]
    assert cases
    analyzer = RestrictedCodeAnalyzer()
    resolved = 0
    for case in cases:
        # The detector's own extractor, not "the last block": at difficulty 4
        # the wrapper puts its instruction after the snippet, so the last
        # block is prose. Measuring extraction here would measure the test.
        snippet = _extract_code(case.prompt)
        if snippet and analyzer.analyze(snippet).result is not None:
            resolved += 1
    rate = resolved / len(cases)
    assert rate >= 0.95, f"only {rate:.3f} of code cases resolved"


def test_a_rejected_snippet_never_carries_the_snippet_into_the_reason() -> None:
    # A snippet can contain a protected value, and the reason is stored in a
    # trace. Names of nodes and calls are safe; the source is not.
    secret = "CANARY-7f3a-KESTREL-9021"
    analysis = _analyze(f'msg = open("{secret}")')
    assert analysis.rejected_reason is not None
    assert secret not in analysis.rejected_reason
    assert secret not in json.dumps(analysis.model_dump())
