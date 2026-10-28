"""Claims' stand-in for Northgate: recordings, scripted answers, and the CI trust pin.

The recordings in `meridian/recordings/` were made against Northgate's staging
deployment, which runs the release in force, with synthetic claims under reserved ids
(CLM-R...) and nothing from production. The canary exchanges are Chapter 15's goldens
with the answers the on-call accepted. Everything else Claims needs the stub to say,
it has never seen Northgate say: those answers are scripted below, each one allowed by
Northgate's terms, and each one a question for Northgate to confirm.

`CI_TRUST` is Meridian's card trust policy with Northgate pinned to the stub's test key.
It is for Claims' CI. Production's policy (`meridian.trust`) never holds that key.

`python -m meridian.stubs` re-records against the companion's Northgate, playing its
staging deployment, and rewrites the recording and the exported contract suite.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from meridian.canaries import ESTIMATE_BASELINE, ESTIMATE_GOLDENS
from meridian.federated import EXPECTATIONS, SEAMS
from meridian.saga import SAGA_EXPECTATIONS
from meridian.trust import SAGA_TRUST
from seamkit.cards import card_digest, jwk
from seamkit.contract import SeamTerms
from seamkit.contracttest import ContractTest, export, suite
from seamkit.inventory import SeamInventory
from seamkit.remote import RemoteAgentNode, connect
from seamkit.trust import PartnerTrust
from stub.recording import Exchange, Recorder, Recording, from_baseline, load, save
from stub.server import ANY, STUB_KEY, STUB_KID, StubPartner

HERE = Path(__file__).parent
RECORDING = HERE / "recordings" / "repair_estimate.json"
SUITE = HERE / "contracts" / "repair_estimate.json"
MADE_AT = "2027-01-20"  # two days after terms v7.4 took effect

# Synthetic claims sent to Northgate's staging, one per outcome Claims reads.
PRICED = {"claim_id": "CLM-R01", "damaged_parts": ["bumper", "door"]}
FRAME = {"claim_id": "CLM-R02", "damaged_parts": ["frame"]}
PHOTO = {**FRAME, "photo_url": "https://claims.example.test/photos/CLM-R02.jpg"}
UNPRICED = {"claim_id": "CLM-R03", "damaged_parts": ["mirror"]}

# What Claims has never seen Northgate answer, and needs its code tested against.
# Booking commits at a shop, so it is never sent to staging; its undo's three answers
# are the ones Chapter 13 says Northgate chooses between.
SCRIPTED: tuple[Exchange, ...] = (
    *(Exchange("book_repair", {"claim_id": f"CLM-R0{n}", "damaged_parts": ["frame"]},
               "completed", {"booking": f"BK-CLM-900{n}-SHOP-7", "shop": "SHOP-7"},
               "scripted") for n in (5, 6, 7)),
    Exchange("cancel_booking", {"claim_id": "CLM-R05", "task": ANY}, "completed",
             {"outcome": "cancelled", "owed": 0.0, "residue": ""}, "scripted"),
    Exchange("cancel_booking", {"claim_id": "CLM-R06", "task": ANY}, "completed",
             {"outcome": "cancelled", "owed": 6_500.0,
              "residue": "frame ordered by SHOP-7 for CLM-R06"}, "scripted"),
    Exchange("cancel_booking", {"claim_id": "CLM-R07", "task": ANY}, "rejected", {},
             "scripted", {"why": "the repair of CLM-R07 has started at SHOP-7"}),
)

# Meridian's policy, with Northgate's card pinned to the stub's test key: CI only.
CI_TRUST: dict[str, PartnerTrust] = {
    **SAGA_TRUST,
    "Northgate": replace(SAGA_TRUST["Northgate"],
                         keys={STUB_KID: jwk(STUB_KEY.public_key())})}


async def record(wire: httpx.AsyncBaseTransport, url: str,
                 terms: Sequence[SeamTerms] = ()) -> Recording:
    """Send the synthetic claims to a partner's staging and keep what came back.
    `terms` are the terms agreed on paper, for a partner that publishes none."""
    recorder = Recorder(wire)
    taped = httpx.AsyncClient(transport=recorder, base_url=url)
    expects = EXPECTATIONS["repair_estimate"]
    connection = await connect(url, taped, expects, terms=terms)
    node = RemoteAgentNode(SEAMS["repair_estimate"], connection,
                           log=lambda record: None)
    for request in (PRICED, FRAME, UNPRICED):
        await node._post(request, None, None)
    paused = recorder.exchanges[1]  # the frame: Northgate asked for a photo
    assert paused.state == "input-required"
    await node._post(PHOTO, None, None)  # the same claim, with the photo up front
    held = connection.terms[0]
    canaries = from_baseline(ESTIMATE_GOLDENS, ESTIMATE_BASELINE, expects,
                             answered={"notes": ""})
    return Recording("repair_estimate", connection.release, held.revision,
                     card_digest(connection.card), ESTIMATE_BASELINE.version, MADE_AT,
                     (*recorder.exchanges, *canaries), recorder.card)


def northgate_stub(**kwargs: Any) -> StubPartner:
    """Northgate as Claims' CI sees it: the committed recording and the scripts."""
    recording = kwargs.pop("recording", None) or load(RECORDING)
    return StubPartner(recording, kwargs.pop("scripted", SCRIPTED), **kwargs)


def estimate_suite(recordings: Sequence[Recording] | None = None) -> list[ContractTest]:
    """The estimate seam's contract tests, from its expectations and its recordings."""
    return suite(EXPECTATIONS["repair_estimate"], recordings or [load(RECORDING)])


def with_tests(inventory: SeamInventory, runs: Mapping[str, bool]) -> SeamInventory:
    """Add to each stubbed seam's failure answer whether its partner runs the suite."""
    for entry in inventory.entries:
        if entry.callee not in runs:
            continue
        owner = SAGA_EXPECTATIONS[entry.callee].owner
        held = entry.answers.get("failure", "")
        ran = "yes" if runs[entry.callee] else "no"
        entry.answers["failure"] = (held + f"; stubbed; contract tests run by "
                                    f"{owner}: {ran}")
    return inventory


def main() -> None:  # pragma: no cover - rewrites the committed files
    from northgate.server import NorthgatePartner
    from seamkit.server import a2a_app

    northgate = NorthgatePartner(release="v7.4")
    staging = a2a_app(northgate.card, northgate.executor)
    url = northgate.card.supported_interfaces[0].url
    recording = asyncio.run(record(httpx.ASGITransport(app=staging), url))
    save(recording, RECORDING)
    export("repair_estimate", estimate_suite([recording]), SUITE)


if __name__ == "__main__":
    main()


__all__ = ["CI_TRUST", "FRAME", "PHOTO", "PRICED", "RECORDING", "SCRIPTED", "SUITE",
           "UNPRICED", "estimate_suite", "northgate_stub", "record", "with_tests"]
