"""Standard base64 transformation."""

import base64
import random

from tipguard.benchmark.transformations.base import Encoded, Family


class Base64Transformation:
    family = Family.BASE64
    deterministic = True

    def encode(self, text: str, rng: random.Random) -> Encoded:
        del rng
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return Encoded(payload=payload, family=self.family, hint="base64")

    def decode(self, encoded: Encoded) -> str:
        return base64.b64decode(encoded.payload).decode("utf-8")
