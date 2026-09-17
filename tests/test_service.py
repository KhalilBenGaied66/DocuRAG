from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.outputs import ChatResult

from docurag.models import ChatTurn, User
from docurag.service import RAGService, SentinelGate
from tests.conftest import FakeRepo, FakeRetriever, make_hit, section

USER = User(id="u1", groups=["maintenance"])


class ExplodingChat(BaseChatModel):
    """Fails the test if the answer path calls the LLM when it should not."""

    @property
    def _llm_type(self) -> str:
        return "exploding"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        raise AssertionError("the LLM must not be called")


def build(settings, *, hits, chat, fast=None, sections=None, reranker="none"):
    repo = FakeRepo(sections if sections is not None else {10: section("Serrer à 25 N·m.")})
    retriever = FakeRetriever(hits, reranker=reranker)
    service = RAGService(settings, repo, retriever, chat, fast or ExplodingChat())
    return service, repo, retriever


async def collect(service, question, **kwargs):
    events = [e async for e in service.stream(question, USER, **kwargs)]
    tokens = "".join(e.data for e in events if e.type == "token")
    return events, tokens, events[-1].data


async def test_answer_keeps_valid_citations_and_strips_invented_ones(settings):
    chat = FakeListChatModel(responses=["Le couple est de 25 N·m [S1]. Voir [S7]."])
    service, repo, _ = build(settings, hits=[make_hit()], chat=chat)

    answer = await service.answer("Couple des vis M8 ?", USER)

    assert not answer.abstained
    assert "[S1]" in answer.text and "[S7]" not in answer.text
    assert answer.invalid_citations == ["S7"]
    assert [c.source_id for c in answer.citations] == ["S1"]
    assert (answer.citations[0].page_start, answer.citations[0].page_end) == (12, 13)
    assert repo.logged[0]["question"] is None  # questions are not stored by default
    assert repo.logged[0]["invalid_citations"] == 1


async def test_sentinel_never_reaches_the_user(settings):
    service, _, _ = build(settings, hits=[make_hit()], chat=FakeListChatModel(responses=["NO_ANSWER"]))

    events, tokens, answer = await collect(service, "Quelle est la couleur du ciel ?")

    assert answer.abstained
    assert "NO_ANSWER" not in tokens
    assert tokens == settings.abstain_message == answer.text
    assert [e.type for e in events][0] == "sources" and events[-1].type == "done"


async def test_low_rerank_score_abstains_without_calling_the_llm(settings):
    settings = settings.model_copy(update={"abstain_threshold": 0.8})
    hit = make_hit(rerank=0.3)
    service, _, _ = build(settings, hits=[hit], chat=ExplodingChat(), reranker="cross-encoder")

    answer = await service.answer("Question hors sujet", USER)

    assert answer.abstained and answer.top_score == 0.3
    assert [s.source_uri for s in answer.sources] == [hit.source_uri]  # closest sources shown


async def test_threshold_is_ignored_without_a_reranker(settings):
    settings = settings.model_copy(update={"abstain_threshold": 0.8})
    chat = FakeListChatModel(responses=["Réponse [S1]"])
    service, _, _ = build(settings, hits=[make_hit(score=0.03)], chat=chat, reranker="none")

    answer = await service.answer("Couple ?", USER)

    assert not answer.abstained  # RRF scores are not on the reranker's scale


async def test_no_hits_abstains(settings):
    service, _, _ = build(settings, hits=[], chat=ExplodingChat())
    answer = await service.answer("Couple ?", USER)
    assert answer.abstained and answer.sources == []


async def test_follow_up_is_rewritten_before_retrieval(settings):
    fast = FakeListChatModel(responses=["« Quel est le couple de serrage sur la P-300 ? »"])
    chat = FakeListChatModel(responses=["25 N·m [S1]"])
    service, _, retriever = build(settings, hits=[make_hit()], chat=chat, fast=fast)

    answer = await service.answer(
        "et pour la P-300 ?", USER,
        history=[ChatTurn(role="user", content="Couple de serrage sur la P-200 ?"),
                 ChatTurn(role="assistant", content="25 N·m [S1]")],
    )

    assert retriever.queries[0] == ("Quel est le couple de serrage sur la P-300 ?", ["maintenance"])
    assert answer.standalone_question.startswith("Quel est")
    assert "rewrite_ms" in answer.timings_ms


async def test_streamed_tokens_equal_the_final_text(settings):
    raw = "Étapes :\n1. Consigner la pompe [S1]\n2. Serrer à 25 N·m [S1]"
    service, _, _ = build(settings, hits=[make_hit()], chat=FakeListChatModel(responses=[raw]))

    _, tokens, answer = await collect(service, "Procédure ?")

    assert tokens == raw == answer.text
    assert {"first_token_ms", "generation_ms", "total_ms"} <= answer.timings_ms.keys()


async def test_contexts_only_returned_on_request(settings):
    chat = FakeListChatModel(responses=["x [S1]", "x [S1]"])
    service, _, _ = build(settings, hits=[make_hit()], chat=chat)
    assert (await service.answer("q ?", USER)).contexts is None
    assert (await service.answer("q ?", USER, include_context=True)).contexts == ["Serrer à 25 N·m."]


@pytest.mark.parametrize(
    ("pieces", "abstained", "emitted"),
    [
        (["NO", "_ANS", "WER"], True, ""),
        ([" ", "NO_ANSWER."], True, ""),
        (["N", "O", " problème"], False, "NO problème"),
        (["No", "n, le couple…"], False, "Non, le couple…"),
        (["NO_A"], True, ""),  # stream ended on a sentinel prefix
        (["Le ", "couple"], False, "Le couple"),
    ],
)
def test_sentinel_gate(pieces, abstained, emitted):
    gate = SentinelGate()
    out = [s for p in pieces for s in gate.feed(p)] + gate.close()
    assert gate.abstained is abstained
    assert "".join(out) == emitted
