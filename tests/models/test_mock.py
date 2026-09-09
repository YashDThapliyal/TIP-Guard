import re

import pytest

from tipguard.models.mock import MockProvider, MockRule
from tipguard.models.types import Message, ModelRequest


def test_simple_builds_messages() -> None:
    req = ModelRequest.simple("hi", system="sys")
    assert req.messages == (
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
    )


def test_mock_echoes_by_default() -> None:
    resp = MockProvider().complete(ModelRequest.simple("hello world"))
    assert resp.text == "ECHO: hello world"
    assert resp.provider == "mock"
    assert resp.output_tokens == 3
    assert resp.cached is False


def test_mock_applies_first_matching_rule() -> None:
    provider = MockProvider(rules=[MockRule(r"canary", "CANARY-1"), MockRule(r".*", "other")])
    assert provider.complete(ModelRequest.simple("Tell me the Canary")).text == "CANARY-1"
    assert provider.complete(ModelRequest.simple("weather")).text == "other"


def test_mock_default_text() -> None:
    assert (
        MockProvider(default="I cannot help.").complete(ModelRequest.simple("x")).text
        == "I cannot help."
    )


def test_mock_rule_rejects_an_invalid_pattern_at_construction() -> None:
    # A bad pattern must fail where the rule is written, not on the first
    # prompt that happens to reach the provider mid-run.
    with pytest.raises(re.error):
        MockRule("(unclosed", "never")


def test_mock_rule_compares_by_pattern_and_response_only() -> None:
    assert MockRule("a", "b") == MockRule("a", "b")
    assert MockRule("a", "b") != MockRule("a", "c")
