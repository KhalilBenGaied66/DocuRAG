from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from maintagent.erp import ErpClient
from maintagent.graph import heuristic_intent
from maintagent.mock_erp import create_mock_erp
from maintagent.tools import TOOLS, EquipmentHistoryArgs, PartsStockArgs, tool_schemas


def erp_with(handler) -> ErpClient:
    return ErpClient("http://erp.test", retries=2, backoff_s=0, transport=httpx.MockTransport(handler))


async def test_erp_retries_transient_errors():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"items": [{"ref": "GM-45", "available": 2}], "unknown": ["XX"]})

    erp = erp_with(handler)
    stock = await erp.stock(["GM-45", "XX"])
    await erp.aclose()

    assert len(attempts) == 2
    assert stock == {"GM-45": {"ref": "GM-45", "available": 2}, "XX": None}
    assert attempts[0].url.params["refs"] == "GM-45,XX"


async def test_erp_gives_up_after_bounded_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    erp = erp_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await erp.stock(["GM-45"])
    await erp.aclose()
    assert len(calls) == 3  # first try + 2 retries


async def test_erp_timeouts_are_retried_then_raised():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    erp = erp_with(handler)
    with pytest.raises(httpx.ReadTimeout):
        await erp.stock(["GM-45"])
    await erp.aclose()


async def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400)

    erp = erp_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await erp.stock(["GM-45"])
    await erp.aclose()
    assert len(calls) == 1


def test_mock_erp_serves_known_and_unknown_refs():
    with TestClient(create_mock_erp()) as client:
        body = client.get("/v1/stock", params={"refs": "GM-45,XX-1"}).json()
    assert [i["ref"] for i in body["items"]] == ["GM-45"] and body["unknown"] == ["XX-1"]
    with TestClient(create_mock_erp(failure_rate=1.0)) as client:
        assert client.get("/v1/stock", params={"refs": "GM-45"}).status_code == 503


def test_tool_schemas_are_strict():
    schemas = {s["function"]["name"]: s["function"] for s in tool_schemas()}
    assert set(schemas) == set(TOOLS) == {"search_docs", "equipment_history", "parts_stock"}
    for fn in schemas.values():
        assert fn["description"] and fn["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("model", "args"),
    [
        (EquipmentHistoryArgs, {"equipment_tag": "P-200-07; DROP TABLE equipment"}),
        (EquipmentHistoryArgs, {"equipment_tag": "P-200-07", "since_days": 100_000}),
        (EquipmentHistoryArgs, {"equipment_tag": "E-41"}),  # an alarm code, not equipment
        (PartsStockArgs, {"part_refs": []}),
        (PartsStockArgs, {"part_refs": ["GM-45' OR 1=1"]}),
        (PartsStockArgs, {"part_refs": ["GM-45"], "site": "all"}),
    ],
)
def test_tool_arguments_are_validated(model, args):
    with pytest.raises(ValidationError):
        model.model_validate(args)


@pytest.mark.parametrize(
    ("report", "kind", "tag"),
    [
        ("Pompe P-200-07 : alarme E-41", "incident", "P-200-07"),
        ("Fuite sur la P-2041 depuis hier", "incident", "P-2041"),
        ("Alarme E-12 sur la P-200-07, palier très chaud", "incident", "P-200-07"),
        ("Que signifie le code E-57 ?", "question", None),
        ("Quel couple de serrage pour le carter ?", "question", None),
        ("Documentation de la P-200-07 ?", "question", "P-200-07"),
    ],
)
def test_rule_based_classification_fallback(report, kind, tag):
    intent = heuristic_intent(report)
    assert (intent.kind, intent.equipment_tag) == (kind, tag)
