"""The task-state-to-outcome mapping is total; only a completed task is an answer."""

import pytest
from a2a.types import Artifact, Task, TaskState, TaskStatus

from seamkit.errors import MalformedAnswer, PartnerFailed, SeamError, TaskUnfinished
from seamkit.remote import OUTCOMES, RemoteAgentNode, SeamSpec


def test_every_state_the_spec_defines_has_exactly_one_outcome() -> None:
    # A state a later SDK adds fails this test instead of falling through.
    assert set(OUTCOMES) == set(TaskState.values())


def test_only_a_completed_task_carries_an_answer() -> None:
    answers = [s for s, error in OUTCOMES.items() if error is None]
    assert answers == [TaskState.TASK_STATE_COMPLETED]
    assert all(issubclass(e, SeamError) for e in OUTCOMES.values() if e is not None)


def node() -> RemoteAgentNode:
    seam = SeamSpec("fraud_screening", "Risk", "screen_fraud", ("claim_id",),
                    {"fraud_score": "fraud_score"})
    return RemoteAgentNode(seam, partner=None)  # type: ignore[arg-type]


def task(state: int, data: dict[str, object] | None = None) -> Task:
    from a2a.helpers import new_data_part

    status = TaskStatus(state=state)  # type: ignore[arg-type]
    t = Task(id="t-1", context_id="c-1", status=status)
    if data is not None:
        t.artifacts.append(Artifact(artifact_id="a-1", parts=[new_data_part(data)]))
    return t


def test_working_is_never_read_as_an_answer() -> None:
    partial = task(TaskState.TASK_STATE_WORKING, {"fraud_score": 12.0})
    with pytest.raises(TaskUnfinished):
        node()._outcome(partial)


def test_canceled_by_the_partner_is_a_failure() -> None:
    with pytest.raises(PartnerFailed):
        node()._outcome(task(TaskState.TASK_STATE_CANCELED))


def test_the_field_map_keeps_only_what_the_seam_names() -> None:
    done = task(TaskState.TASK_STATE_COMPLETED,
                {"fraud_score": 82.0, "model": "fraud-v9", "decision": "paid"})
    assert node()._outcome(done) == {"fraud_score": 82.0}  # no "decision" from Risk


def test_a_completed_task_without_the_field_is_malformed() -> None:
    with pytest.raises(MalformedAnswer):
        node()._outcome(task(TaskState.TASK_STATE_COMPLETED, {"score": 0.82}))
