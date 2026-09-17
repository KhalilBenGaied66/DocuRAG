"""Command line: docurag init-db | ingest | sync | ask | worker | serve | eval."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from docurag.config import Settings, get_settings
from docurag.db.pool import SchemaMismatch
from docurag.logging_setup import setup_logging


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    if sys.platform == "win32":
        # psycopg's async driver needs a selector loop; Windows defaults to proactor.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def _groups(value: str) -> list[str]:
    groups = [g.strip() for g in value.split(",") if g.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("at least one group is required")
    return groups


def _print_result(result: Any) -> None:
    line = f"{result.status:<12} {result.source_uri}"
    if result.error:
        line += f"  — {result.error}"
    print(line)
    report = result.report or {}
    if report:
        keys = ("pages", "sections", "chunks", "table_chunks", "tokens_p50", "tokens_p95",
                "embeddings_computed", "embeddings_cached", "duration_ms")
        print("             " + ", ".join(f"{k}={report[k]}" for k in keys if k in report))
        for warning in report.get("warnings", []):
            print(f"             warning: {warning}")


async def _with_container(settings: Settings, fn: Any) -> Any:
    from docurag.container import build_container

    container = await build_container(settings)
    try:
        return await fn(container)
    finally:
        await container.close()


async def _cmd_init_db(args: argparse.Namespace, settings: Settings) -> int:
    async def run(container: Any) -> int:
        print(
            f"schema ready: embedding_model={settings.embedding_model}"
            f" dimensions={settings.embedding_dimensions}"
            f" text_search={settings.text_search_config}"
            f" iterative_scan={container.repo.iterative_scan}"
        )
        return 0

    return await _with_container(settings, run)


async def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from docurag.models import DocumentSpec

    if args.doc_key and len(args.paths) > 1:
        print("error: --doc-key applies to a single file", file=sys.stderr)
        return 2

    async def run(container: Any) -> int:
        failed = 0
        for path in args.paths:
            spec = DocumentSpec(
                source_uri="file://" + path.resolve().as_posix(), path=str(path.resolve()),
                title=args.title, doc_key=args.doc_key, version=args.version,
                acl_groups=args.acl,
            )
            result = await container.pipeline.ingest(spec, force=args.force)
            _print_result(result)
            failed += result.status == "failed"
        return 1 if failed else 0

    return await _with_container(settings, run)


async def _cmd_sync(args: argparse.Namespace, settings: Settings) -> int:
    from docurag.ingestion.manifest import load_manifest

    manifest = load_manifest(args.manifest)

    async def run(container: Any) -> int:
        results, removed = await container.pipeline.sync(manifest, prune=args.prune)
        for result in results:
            _print_result(result)
        for uri in removed:
            print(f"{'deleted':<12} {uri}")
        counts: dict[str, int] = {}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) +
              f", deleted: {len(removed)}")
        return 1 if counts.get("failed") else 0

    return await _with_container(settings, run)


async def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from docurag.models import User

    async def run(container: Any) -> int:
        answer = await container.service.answer(
            args.question, User(id="cli", groups=args.groups), mode=args.mode
        )
        print(answer.text, "\n")
        for c in answer.citations:
            pages = ""
            if c.page_start is not None:
                pages = f", p. {c.page_start}" + (
                    f"–{c.page_end}" if c.page_end not in (None, c.page_start) else ""
                )
            print(f"[{c.source_id}] {c.title} — {c.section_path or '(début)'}{pages}")
        print(f"\nabstained={answer.abstained} top_score={answer.top_score} "
              f"tokens={answer.usage.input_tokens}+{answer.usage.output_tokens}")
        print("timings_ms=" + json.dumps(answer.timings_ms))
        return 0

    return await _with_container(settings, run)


async def _cmd_worker(args: argparse.Namespace, settings: Settings) -> int:
    from docurag.worker import run_worker

    async def run(container: Any) -> int:
        processed = await run_worker(container, once=args.once)
        print(f"processed {processed} job(s)")
        return 0

    return await _with_container(settings, run)


async def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from docurag.evaluation.dataset import load_dataset
    from docurag.evaluation.runner import format_report, run_eval, save_result
    from docurag.providers import build_chat_model

    items = load_dataset(args.dataset)[: args.limit or None]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None

    async def run(container: Any) -> int:
        judge = None
        if args.ragas:
            if args.judge_model:
                judge = build_chat_model(settings.model_copy(update={"chat_model": args.judge_model}))
            else:
                print("note: the judge is the generator model; prefer --judge-model", file=sys.stderr)
        result = await run_eval(
            container, items, label=args.label, mode=args.mode, default_groups=args.groups,
            generation=not args.retrieval_only, ragas=args.ragas, judge=judge,
            calibrate=args.calibrate,
        )
        path = save_result(result, args.out)
        print(format_report(result, baseline))
        print(f"\nsaved {path}")
        return 0

    return await _with_container(settings, run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docurag", description="RAG over technical documents")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the setup (config, database, models) and explain fixes")
    p.add_argument("--skip-models", action="store_true", help="don't call the models")

    sub.add_parser("init-db", help="create the schema and check it matches the settings")

    p = sub.add_parser("ingest", help="ingest files (incremental: unchanged files are skipped)")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--acl", type=_groups, required=True, help="comma-separated reader groups")
    p.add_argument("--doc-key")
    p.add_argument("--version")
    p.add_argument("--title")
    p.add_argument("--force", action="store_true", help="re-process even if unchanged")

    p = sub.add_parser("sync", help="bring the index in line with a corpus manifest")
    p.add_argument("manifest", type=Path)
    p.add_argument("--prune", action="store_true", help="delete documents no longer listed")

    p = sub.add_parser("ask", help="ask a question as a given set of groups")
    p.add_argument("question")
    p.add_argument("--groups", type=_groups, required=True)
    p.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")

    p = sub.add_parser("worker", help="process queued ingestion jobs")
    p.add_argument("--once", action="store_true", help="exit when the queue is empty")

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("eval", help="score retrieval and answers on an eval set")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--groups", type=_groups, required=True, help="default groups per question")
    p.add_argument("--label", default="run")
    p.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    p.add_argument("--baseline", type=Path, help="previous result JSON to compare with")
    p.add_argument("--ragas", action="store_true", help="add RAGAS metrics (eval extra)")
    p.add_argument("--judge-model", help="chat deployment used as RAGAS judge")
    p.add_argument("--calibrate", action="store_true", help="suggest an abstention threshold")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, default=Path("eval/results"))
    return parser


def configure_console() -> None:
    # A redirected Windows console uses a legacy code page: never crash on "→" or "≥".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def report_startup_error(exc: BaseException) -> int | None:
    """Readable messages for the errors people hit while setting up; None if not one of them."""
    import psycopg
    from psycopg_pool import PoolTimeout
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        print("configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(map(str, error["loc"])) or "settings"
            print(f"  - {field}: {error['msg'].removeprefix('Value error, ')}", file=sys.stderr)
    elif isinstance(exc, PoolTimeout | psycopg.OperationalError):
        print(f"error: cannot reach the database ({' '.join(str(exc).split())[:200]})", file=sys.stderr)
    elif isinstance(exc, SchemaMismatch | RuntimeError | ValueError):
        print(f"error: {exc}", file=sys.stderr)
    else:
        return None
    print("run `docurag doctor` for a full check of the setup", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_console()
    long_running = args.command in {"serve", "worker"}
    # One-shot commands print their own results: only warnings and errors go to the log.
    setup_logging("INFO" if long_running else "WARNING", json_logs=long_running)
    if args.command == "doctor":
        from docurag.doctor import doctor_main

        return run_async(doctor_main(Settings, check_model_calls=not args.skip_models))

    try:
        settings = get_settings()
        if getattr(args, "no_rerank", False):
            settings = settings.model_copy(update={"reranker": "none"})
        if args.command == "serve":
            import uvicorn

            from docurag.api.app import create_app

            config = uvicorn.Config(
                create_app(), host=args.host, port=args.port, log_config=None, proxy_headers=True
            )
            run_async(uvicorn.Server(config).serve())
            return 0
        handler = {
            "init-db": _cmd_init_db, "ingest": _cmd_ingest, "sync": _cmd_sync, "ask": _cmd_ask,
            "worker": _cmd_worker, "eval": _cmd_eval,
        }[args.command]
        return run_async(handler(args, settings))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        code = report_startup_error(exc)
        if code is None:
            raise
        return code


if __name__ == "__main__":
    raise SystemExit(main())
