"""The pydantic base class every TIP-Guard model derives from."""

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable and closed: attribute assignment raises, unknown keys raise.

    `frozen=True` is the project's immutability rule expressed in the type:
    a derived value is a new object from `model_copy(update=...)`, never a
    mutated one. `extra="forbid"` is the input-validation rule: a record
    replayed from an older run, or a hand-written config, that carries a
    field this version no longer knows about fails loudly at the boundary
    instead of being silently dropped and analysed as if it had never
    existed. Both are wanted on every model here, so they live in one place
    rather than being repeated (and, as the foundation review found,
    unevenly) on each `model_config`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
