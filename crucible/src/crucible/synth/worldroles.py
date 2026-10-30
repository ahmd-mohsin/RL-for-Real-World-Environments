"""The roles of World-family synthesis, gathered for the wire (``factory/synth_wire.py``) and the family checks.

| role | module | stage | sees | differs from |
|---|---|---|---|---|
| ``domain_cataloguer`` | ``catalogue`` | W0 | a function area, the catalogue so far | — |
| ``domain_spec_writer`` | ``domain`` | W0 | the keyword, its function | — |
| ``family_designer`` | ``families`` | W1 | the domain spec, the realism surface | — |
| ``data_spec_writer`` | ``families`` | W2 | spec, design, schema, the generator format, the last sample's problems | — |
| ``doc_writer`` | ``families`` | W3 | spec, tool names, policy parameters, data notes | — |
| ``actor_writer`` | ``families`` | W4 | spec, table fields, document titles | — |
| ``realism_rater`` | ``worldgates`` | W5 | an instance summary, the rubric (≥ 2 families, independent) | every World builder role |

The F1 roles of ``roles.py`` (schema deriver, App coder, blind tester, debugger, test approver) are reused unchanged;
the family builder shows them the design together with the policy parameters and the platform notes.
"""

from __future__ import annotations

from crucible.synth.catalogue import DOMAIN_CATALOGUER
from crucible.synth.domain import DOMAIN_SPEC_WRITER
from crucible.synth.families import WORLD_BUILD_ROLES
from crucible.synth.roles import ROLES, Role
from crucible.synth.worldgates import REALISM_RATER

WORLD_ROLES: dict[str, Role] = {r.name: r for r in (DOMAIN_CATALOGUER, DOMAIN_SPEC_WRITER, *WORLD_BUILD_ROLES.values(), REALISM_RATER)}
ALL_ROLES: dict[str, Role] = {**ROLES, **WORLD_ROLES}

__all__ = ["ALL_ROLES", "WORLD_ROLES"]
