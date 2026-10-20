"""Model clients behind LangChain interfaces: Azure OpenAI, OpenAI, or offline fakes."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from docurag.config import Settings

_WORD = re.compile(r"\w+", re.UNICODE)
_TAG = re.compile(r"\b[A-Z]{1,4}-\d{3,5}(?:-\d{1,3})?\b")  # equipment tags, not alarm codes (E-12)
_INCIDENT = re.compile(r"alarme|panne|fuite|vibr|bruit|défaut|arrêt|échauff|\bE-\d{2}\b", re.I)
_SOURCE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.S)


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors (feature hashing). Offline tests only, never prod."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(digest, "little") % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0  # a zero vector has no cosine distance
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ExtractiveFakeChatModel(BaseChatModel):
    """Deterministic stand-in for an LLM, so the whole system runs offline (plumbing, not quality).

    - RAG answers: the first sentence of source S1, cited; NO_ANSWER without sources.
    - Structured output and tools (MaintAgent): rule-based classification, one round of
      search_docs + equipment_history, then a low-confidence draft citing what was found.
    """

    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "docurag-extractive-fake"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> ExtractiveFakeChatModel:
        names = [convert_to_openai_tool(tool)["function"]["name"] for tool in tools]
        return self.model_copy(update={"bound_tools": names})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.bound_tools:
            return ChatResult(generations=[ChatGeneration(message=self._tool_turn(messages))])
        prompt = str(messages[-1].content)
        sources = _SOURCE.findall(prompt)
        if "SOURCES" not in prompt:  # query rewriting: echo the last question
            text = prompt.rsplit(":", 1)[-1].strip()
        elif not sources:
            text = "NO_ANSWER"
        else:
            sid, body = sources[0]
            first = re.split(r"(?<=[.!?])\s", body.strip(), maxsplit=1)[0]
            text = f"{first[:400]} [{sid}]"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _tool_turn(self, messages: list[BaseMessage]) -> AIMessage:
        report = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        found = _TAG.search(report)
        tag = found.group(0) if found else None
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        evidence = _source_ids(observations)
        names = set(self.bound_tools)

        def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {"name": name, "args": args, "id": f"fake-{name}-{len(messages)}", "type": "tool_call"}

        if "Intent" in names:
            kind = "incident" if tag and _INCIDENT.search(report) else "question"
            return AIMessage("", tool_calls=[call("Intent", {"kind": kind, "equipment_tag": tag})])
        if "WorkOrderDraft" in names:
            summary = next(
                (str(m.content) for m in reversed(messages)
                 if isinstance(m, AIMessage) and not m.tool_calls and str(m.content).strip()),
                "Diagnostic non établi.",
            )
            draft = {
                "equipment_tag": tag or "INCONNU", "title": "Diagnostic à confirmer",
                "diagnosis": summary[:2000], "probable_cause": "À confirmer par un technicien",
                "actions": ["Consigner l'équipement", "Contrôler selon la documentation citée"],
                "parts": [], "priority": "P2", "safety_notes": [], "sources": evidence,
                "confidence": "low",
            }
            return AIMessage("", tool_calls=[call("WorkOrderDraft", draft)])
        if "search_docs" in names and not observations:
            calls = [call("search_docs", {"query": report[:300]})]
            if tag:
                calls.append(call("equipment_history", {"equipment_tag": tag}))
            return AIMessage("", tool_calls=calls)
        cited = ", ".join(evidence) if evidence else "aucune source trouvée"
        return AIMessage(f"Diagnostic hors ligne (modèle factice) à partir de : {cited}.")


def _source_ids(observations: list[ToolMessage]) -> list[str]:
    ids: list[str] = []
    for message in observations:
        ids += [s for s in re.findall(r'"source_id": "(doc:\d+)"', str(message.content)) if s not in ids]
    return ids


def _azure_auth(settings: Settings) -> dict[str, Any]:
    if settings.azure_use_entra_id:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise RuntimeError("azure_use_entra_id needs the azure extra") from exc
        provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return {"azure_ad_token_provider": provider}
    if settings.api_key is None:
        raise ValueError("set DOCURAG_API_KEY, or DOCURAG_AZURE_USE_ENTRA_ID=true")
    return {"api_key": settings.api_key}


def _ollama() -> Any:
    try:
        import langchain_ollama
    except ImportError as exc:
        raise RuntimeError("provider=ollama needs the ollama extra: pip install -e '.[ollama]'") from exc
    return langchain_ollama


def _require_azure_endpoint(settings: Settings) -> str:
    if not settings.azure_endpoint:
        raise ValueError("provider=azure needs DOCURAG_AZURE_ENDPOINT")
    return settings.azure_endpoint


def build_chat_model(settings: Settings, *, fast: bool = False) -> BaseChatModel:
    name = settings.fast_model if fast else settings.chat_model
    common: dict[str, Any] = {
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
        "stream_usage": True,
    }
    if settings.chat_temperature is not None:
        common["temperature"] = settings.chat_temperature

    if settings.provider == "fake":
        return ExtractiveFakeChatModel()
    if settings.provider == "ollama":
        return _ollama().ChatOllama(
            model=name,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            reasoning=settings.ollama_reasoning,
            temperature=settings.chat_temperature,
            client_kwargs={"timeout": settings.llm_timeout_s},
        )
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, api_key=settings.api_key, **common)
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=name,
        azure_endpoint=_require_azure_endpoint(settings),
        api_version=settings.azure_api_version,
        **_azure_auth(settings),
        **common,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.provider == "fake":
        return HashingEmbeddings(settings.embedding_dimensions)
    if settings.provider == "ollama":
        return _ollama().OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
    extra: dict[str, Any] = {"max_retries": settings.llm_max_retries}
    if "embedding-3" in settings.embedding_model:  # only text-embedding-3-* accept `dimensions`
        extra["dimensions"] = settings.embedding_dimensions
    if settings.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.api_key, **extra)
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_deployment=settings.embedding_model,
        azure_endpoint=_require_azure_endpoint(settings),
        api_version=settings.azure_api_version,
        **_azure_auth(settings),
        **extra,
    )


def content_text(message: BaseMessage) -> str:
    """Text of a message or chunk, whether content is a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        if not isinstance(block, dict) or block.get("type") in (None, "text")
    )
