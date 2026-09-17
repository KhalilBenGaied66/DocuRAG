"""Command line: maintagent seed | erp | submit | decide | show | resume | serve | eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from docurag.cli import configure_console, report_startup_error, run_async
from docurag.logging_setup import setup_logging
from docurag.models import User
from maintagent.models import AgentPermissionError, TicketStateError


def _groups(value: str) -> list[str]:
    return [g.strip() for g in value.split(",") if g.strip()]


def _print_view(view: Any) -> None:
    print(f"ticket {view.ticket_id}: {view.status}  (intent={view.intent}, equipment="
          f"{view.equipment_tag}, steps={view.steps}, tokens={view.tokens})")
    for call in view.tool_calls:
        mark = "ok " if call["ok"] else "ERR"
        print(f"  [{mark}] {call['tool']} {json.dumps(call.get('args'), ensure_ascii=False)}")
    if view.answer:
        print("\n" + view.answer["text"])
    if view.proposal:
        p = view.proposal
        print(f"\nProposition — {p['title']} [{p['priority']}, confiance {p['confidence']}]")
        print(f"Diagnostic : {p['diagnosis']}\nCause probable : {p['probable_cause']}")
        for i, action in enumerate(p["actions"], start=1):
            print(f"  {i}. {action}")
        for part in p["parts"]:
            print(f"  pièce {part['ref']} x{part['quantity']}")
        print(f"Sources : {', '.join(p['sources']) or '—'}")
    for issue in view.issues:
        print(f"  ! {issue}")
    if view.incomplete:
        print("  ! proposition incomplète (limite atteinte ou sortie invalide)")
    if view.work_order_id:
        print(f"\nOrdre de travail créé : {view.work_order_id}")


async def _with_agent(fn: Any) -> Any:
    from maintagent.container import build_agent_container

    container = await build_agent_container()
    try:
        return await fn(container)
    finally:
        await container.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maintagent", description="Maintenance diagnosis agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed", help="load equipment.csv and interventions.csv")
    p.add_argument("directory", type=Path)

    p = sub.add_parser("erp", help="run the mock ERP")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--failure-rate", type=float, default=0.0)

    for name, help_text in (("submit", "submit a ticket"), ("show", "show a ticket"),
                            ("resume", "resume a stalled ticket"), ("decide", "approve or reject")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("ticket_id")
        if name == "submit":
            p.add_argument("report")
        if name in {"submit", "decide"}:
            p.add_argument("--user", required=True)
            p.add_argument("--groups", type=_groups, required=True)
        if name == "decide":
            choice = p.add_mutually_exclusive_group(required=True)
            choice.add_argument("--approve", action="store_true")
            choice.add_argument("--reject", action="store_true")
            p.add_argument("--feedback")

    p = sub.add_parser("serve", help="run DocuRAG + MaintAgent API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("eval", help="run agent scenarios")
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--user", default="eval-technician")
    p.add_argument("--groups", type=_groups, default=["maintenance"])
    p.add_argument("--approve-as", help="supervisor ID: also approve and check idempotency")
    p.add_argument("--supervisor-groups", type=_groups,
                   default=["maintenance", "maintenance-supervisor"])
    p.add_argument("--label", default="run")
    p.add_argument("--out", type=Path, default=Path("eval/results"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_console()
    long_running = args.command in {"serve", "erp"}
    setup_logging("INFO" if long_running else "WARNING", json_logs=long_running)

    if args.command == "erp":
        import uvicorn

        from maintagent.mock_erp import create_mock_erp

        app = create_mock_erp(failure_rate=args.failure_rate)
        run_async(uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port)).serve())
        return 0
    if args.command == "serve":
        import uvicorn

        from maintagent.app import create_app

        config = uvicorn.Config(create_app(), host=args.host, port=args.port, log_config=None)
        run_async(uvicorn.Server(config).serve())
        return 0

    async def dispatch(container: Any) -> int:
        agent = container.agent
        if args.command == "seed":
            print(await container.business.seed(args.directory))
        elif args.command == "submit":
            requester = User(id=args.user, groups=args.groups)
            _print_view(await agent.submit(args.ticket_id, args.report, requester))
        elif args.command == "decide":
            view = await agent.decide(
                args.ticket_id, User(id=args.user, groups=args.groups),
                approved=args.approve, feedback=args.feedback,
            )
            _print_view(view)
        elif args.command == "show":
            view = await agent.get(args.ticket_id)
            if view is None:
                print("no such ticket", file=sys.stderr)
                return 1
            _print_view(view)
        elif args.command == "resume":
            _print_view(await agent.resume(args.ticket_id))
        elif args.command == "eval":
            from docurag.evaluation.runner import save_result
            from maintagent.evaluation import load_scenarios, run_scenarios

            supervisor = (User(id=args.approve_as, groups=args.supervisor_groups)
                          if args.approve_as else None)
            result = await run_scenarios(
                agent, load_scenarios(args.scenarios),
                requester=User(id=args.user, groups=args.groups), supervisor=supervisor,
                label=f"agent-{args.label}",
            )
            print(json.dumps(result["summary"], indent=2))
            print(f"saved {save_result(result, args.out)}")
        return 0

    try:
        return run_async(_with_agent(dispatch))
    except (AgentPermissionError, TicketStateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        code = report_startup_error(exc)
        if code is None:
            raise
        return code


if __name__ == "__main__":
    raise SystemExit(main())
