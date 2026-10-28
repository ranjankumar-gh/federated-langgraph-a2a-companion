"""The estimate seam's compatibility matrix, as Claims commits it.

Claims keeps every revision of Northgate's terms it has seen, the date it first saw
each one (meridian/seen/), and the matrix rendered from them (meridian/compat/). The
CI test regenerates the table and fails on any cell that differs from the committed
file, so a new revision, a changed expectation, or a new behaviour version reaches a
person as a diff to review before it reaches a claim.
"""

from pathlib import Path

from meridian.canaries import ESTIMATE_BASELINE
from meridian.data_classes import MERIDIAN_POLICY
from meridian.federated import EXPECTATIONS
from northgate.server import (
    NORTHGATE_V7,
    NORTHGATE_V7_1,
    NORTHGATE_V7_2,
    NORTHGATE_V7_3,
    NORTHGATE_V7_4,
    NORTHGATE_V8,
)
from seamkit.compat import load_seen, matrix, render

HERE = Path(__file__).parent
SEEN = HERE / "seen" / "repair_estimate.toml"
COMMITTED = HERE / "compat" / "repair_estimate.md"

# Every revision of Northgate's terms Claims has seen, kept as it was published.
NORTHGATE_REVISIONS = (NORTHGATE_V7, NORTHGATE_V7_1, NORTHGATE_V7_2, NORTHGATE_V8,
                       NORTHGATE_V7_3, NORTHGATE_V7_4)


def estimate_matrix(in_force: str = "v7.4",
                    behaviour: str = f"in band, {ESTIMATE_BASELINE.version}") -> str:
    """The table for the estimate seam, against today's expectations."""
    rows = matrix(NORTHGATE_REVISIONS, EXPECTATIONS["repair_estimate"],
                  MERIDIAN_POLICY, load_seen(SEEN), in_force, behaviour)
    return render("repair_estimate", rows)


if __name__ == "__main__":  # regenerate the committed table after a person reviews it
    COMMITTED.write_text(estimate_matrix(), encoding="utf-8")
