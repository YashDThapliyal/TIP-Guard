"""Tests for tipguard.seeding: deterministic seeding of random generators."""

import random

from tipguard.seeding import seed_everything


def test_seed_everything_is_deterministic() -> None:
    a = seed_everything(7).random()
    b = seed_everything(7).random()
    assert a == b


def test_seed_everything_reseeds_module_random_deterministically() -> None:
    seed_everything(7)
    first = random.random()
    seed_everything(7)
    second = random.random()
    assert first == second
