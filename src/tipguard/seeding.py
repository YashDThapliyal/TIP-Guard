"""Deterministic seeding helpers."""

import random


def seed_everything(seed: int) -> random.Random:
    random.seed(seed)
    return random.Random(seed)
