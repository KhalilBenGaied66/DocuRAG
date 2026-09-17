from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from maintagent.evaluation import Scenario, run_scenarios
from maintagent.models import AgentPermissionError, TicketStateError
from tests.agent_fakes import (
    REPORT,
    SUPERVISOR,
    TECH,
    Harness,
    ai_tools,
    draft,
    incident_script,
    intent,
    tool_call,
)


def incident_harness(**settings) -> Harness:
    return Harness(incident_script(), [intent("incident", "P-200-07")], **settings)


def tool_results(messages) -> dict[str, dict]:
    return {m.tool_call_id: json.loads(m.content) for m in messages if isinstance(m, ToolMessage)}


# ------------------------------------------------------------------ happy path


async def test_incident_runs_to_a_gated_proposal():
    h = incident_harness()
    view = await h.agent.submit("T-1", REPORT, TECH)

    assert view.status == "awaiting_approval"
    assert (view.intent, view.equipment_tag) == ("incident", "P-200-07")
    assert [c["tool"] for c in view.tool_calls] == ["search_docs", "equipment_history", "parts_stock"]
    assert view.proposal["sources"] == ["doc:7"]  # the invented doc:999 was removed …
    assert any("doc:999" in issue for issue in view.issues)  # … and reported
    assert any("6309-2Z" in issue and "0 en stock" in issue for issue in view.issues)
    assert view.work_order_id is None and h.business.create_calls == 0
    assert h.retriever.calls == [("alarme E-41 vibration pompe P-200", ["maintenance"])]
    assert (view.steps, view.tokens) == (3, 500)


async def test_approval_creates_exactly_one_work_order():
    h = incident_harness()
    await h.agent.submit("T-1", REPORT, TECH)

    view = await h.agent.decide("T-1", SUPERVISOR, approved=True)

    assert view.status == "work_order_created" and view.work_order_id == "WO-0001"
    assert view.decision == {"approved": True, "approver": "sup-7", "feedback": None}
    again = await h.agent.submit("T-1", REPORT, TECH)  # same ticket: same run, no model call
    assert again.work_order_id == "WO-0001" and h.business.create_calls == 1
    with pytest.raises(TicketStateError, match="no decision is pending"):
        await h.agent.decide("T-1", SUPERVISOR, approved=True)


async def test_question_goes_to_docurag_without_the_agent_loop():
    h = Harness([], [intent("question")])
    question = "Quel couple de serrage pour le carter de la P-200 ?"

    view = await h.agent.submit("T-2", question, TECH)

    assert view.status == "answered" and view.answer["text"] == "49 N·m [S1]"
    assert h.rag.calls == [(question, ["maintenance"])]
    assert h.chat.calls == [] and view.tokens == 128


# ------------------------------------------------------------------ human in the loop


async def test_approver_must_be_a_supervisor_and_not_the_requester():
    h = incident_harness()
    await h.agent.submit("T-1", REPORT, TECH)
    with pytest.raises(AgentPermissionError, match="supervisor"):
        await h.agent.decide("T-1", TECH, approved=True)

    h2 = incident_harness()
    await h2.agent.submit("T-9", REPORT, SUPERVISOR)
    with pytest.raises(AgentPermissionError, match="requester"):
        await h2.agent.decide("T-9", SUPERVISOR, approved=True)
    assert h.business.create_calls == h2.business.create_calls == 0


async def test_rejection_with_feedback_sends_the_agent_back_to_work():
    script = [*incident_script(), AIMessage("Contrôle du roulement ajouté."),
              draft(title="Contrôle roulement puis réalignement")]
    h = Harness(script, [intent("incident", "P-200-07")])
    await h.agent.submit("T-1", REPORT, TECH)

    view = await h.agent.decide(
        "T-1", SUPERVISOR, approved=False, feedback="Vérifier d'abord le roulement côté accouplement"
    )

    assert view.status == "awaiting_approval"
    assert view.proposal["title"] == "Contrôle roulement puis réalignement"
    reason_call = h.chat.calls[-2]
    assert any(isinstance(m, HumanMessage) and "Vérifier d'abord le roulement" in m.content
               for m in reason_call)
    closed = await h.agent.decide("T-1", SUPERVISOR, approved=False)
    assert closed.status == "rejected" and h.business.create_calls == 0


# ------------------------------------------------------------------ bounds and failures


async def test_step_limit_forces_an_incomplete_proposal_without_dangling_calls():
    loop = [ai_tools(tool_call("search_docs", {"query": f"recherche {i}"}, f"t{i}")) for i in range(2)]
    h = Harness([*loop, draft()], [intent("incident", "P-200-07")], max_steps=2)

    view = await h.agent.submit("T-1", REPORT, TECH)

    assert view.status == "awaiting_approval" and view.incomplete and view.steps == 2
    assert len(view.tool_calls) == 1  # the second batch of calls was never executed
    propose_call = h.chat.calls[-1]
    answered = {m.tool_call_id for m in propose_call if isinstance(m, ToolMessage)}
    for message in propose_call:
        if isinstance(message, AIMessage):
            assert all(tc["id"] in answered for tc in message.tool_calls)


async def test_token_budget_stops_the_loop():
    greedy = [ai_tools(tool_call("search_docs", {"query": "requête"}, "t1"), tokens=5_000)]
    h = Harness([*greedy, draft()], [intent("incident", "P-200-07")], token_budget=1_000)
    view = await h.agent.submit("T-1", REPORT, TECH)
    assert view.incomplete and view.tool_calls == []


async def test_tool_failures_come_back_as_observations():
    bad = ai_tools(
        tool_call("delete_everything", {}, "t1"),
        tool_call("equipment_history", {"equipment_tag": "DROP TABLE"}, "t2"),
        tool_call("parts_stock", {"part_refs": ["GM-45"]}, "t3"),
    )
    h = Harness([bad, AIMessage("Données insuffisantes."), draft(parts=[], sources=[])],
                [intent("incident", "P-200-07")], tool_timeout_s=0.05, erp_delay=0.5)

    view = await h.agent.submit("T-1", REPORT, TECH)

    results = tool_results(h.chat.calls[1])
    assert results["t1"]["error"] == "unknown_tool"
    assert results["t2"]["error"] == "invalid_arguments"
    assert results["t3"]["error"] == "timeout"
    assert view.status == "awaiting_approval"  # the run carried on
    assert "Aucune source documentaire citée" in view.issues


async def test_the_model_cannot_choose_whose_access_it_uses():
    sneaky = ai_tools(tool_call("search_docs", {"query": "grille des salaires", "groups": ["hr"]}, "t1"))
    h = Harness([sneaky, AIMessage("Rien."), draft(sources=[])], [intent("incident", "P-200-07")])

    await h.agent.submit("T-1", REPORT, TECH)

    assert tool_results(h.chat.calls[1])["t1"]["error"] == "invalid_arguments"
    assert h.retriever.calls == []


async def test_parallel_call_cap_still_answers_every_call():
    calls = [tool_call("search_docs", {"query": f"requête {i}"}, f"t{i}") for i in range(6)]
    h = Harness([ai_tools(*calls), AIMessage("ok"), draft()], [intent("incident", "P-200-07")],
                max_parallel_tool_calls=4)

    await h.agent.submit("T-1", REPORT, TECH)

    results = tool_results(h.chat.calls[1])
    assert len(results) == 6 and len(h.retriever.calls) == 4
    assert sum(r.get("error") == "too_many_parallel_calls" for r in results.values()) == 2


async def test_classification_falls_back_to_rules_on_invalid_output():
    h = Harness(incident_script(), [AIMessage("c'est un incident"), AIMessage("incident")])
    view = await h.agent.submit("T-1", REPORT, TECH)
    assert (view.intent, view.equipment_tag) == ("incident", "P-200-07")
    assert len(h.fast.calls) == 2  # one corrective retry, then rules


async def test_invalid_proposal_gets_one_corrective_retry():
    h = Harness([AIMessage("Diagnostic : roulement."), draft(actions=[]), draft()],
                [intent("incident", "P-200-07")])
    view = await h.agent.submit("T-1", REPORT, TECH)
    assert view.proposal["title"] == "Réalignement moteur-pompe" and not view.incomplete
    assert any("Sortie invalide" in str(m.content) for m in h.chat.calls[-1])


async def test_unusable_proposal_becomes_a_minimal_flagged_draft():
    h = Harness([AIMessage("Diagnostic : roulement usé."), AIMessage("pas de JSON"), AIMessage("non")],
                [intent("incident", "P-200-07")])
    view = await h.agent.submit("T-1", REPORT, TECH)
    assert view.incomplete and view.proposal["confidence"] == "low"
    assert view.proposal["diagnosis"] == "Diagnostic : roulement usé."
    assert view.issues[0].startswith("Le modèle n'a pas produit de proposition valide")


async def test_a_crash_resumes_from_the_last_checkpoint():
    h = Harness([RuntimeError("connection reset"), *incident_script()], [intent("incident", "P-200-07")])

    stalled = await h.agent.submit("T-1", REPORT, TECH)
    assert stalled.status == "stalled" and stalled.intent == "incident"

    resumed = await h.agent.resume("T-1")
    assert resumed.status == "awaiting_approval"
    assert len(h.fast.calls) == 1  # classification was not redone


async def test_state_guards():
    h = incident_harness()
    with pytest.raises(TicketStateError):
        await h.agent.resume("nope")
    with pytest.raises(TicketStateError):
        await h.agent.decide("nope", SUPERVISOR, approved=True)
    await h.agent.submit("T-1", REPORT, TECH)
    with pytest.raises(TicketStateError, match="waiting for a supervisor"):
        await h.agent.resume("T-1")


# ------------------------------------------------------------------ scenario evaluation


async def test_scenario_run_scores_trajectories_and_idempotency():
    h = Harness(incident_script(), [intent("incident", "P-200-07"), intent("question")])
    scenarios = [
        Scenario(id="s01", report=REPORT, expected_intent="incident", expected_equipment="P-200-07",
                 expected_tools=["search_docs", "equipment_history"], max_steps=5),
        Scenario(id="s05", report="Quel couple pour le carter ?", expected_intent="question"),
    ]

    result = await run_scenarios(h.agent, scenarios, requester=TECH, supervisor=SUPERVISOR)

    s = result["summary"]
    assert s["task_success"] == s["intent_correct"] == s["tool_recall"] == 1.0
    assert s["write_gated"] == s["work_order_created"] == s["idempotent"] == 1.0
    assert h.business.create_calls == 1


async def test_alarm_code_returned_as_equipment_is_replaced_by_the_real_tag():
    report = "Alarme E-12 sur la P-200-07, palier très chaud"
    h = Harness(incident_script(), [intent("incident", "E-12")])
    view = await h.agent.submit("T-1", report, TECH)
    assert view.equipment_tag == "P-200-07"
