import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_03 = ROOT / "scripts" / "03_generate_candidates_deepseek.py"
SCRIPT_04 = ROOT / "scripts" / "04_validate_candidates.py"
DB_PATH = ROOT / "data" / "raw" / "db.sqlite"
SCHEMA_GUIDE_PATH = ROOT / "assets" / "schema_guide.yaml"
BUSINESS_CONTEXT_PATH = ROOT / "assets" / "schema_business_context.json"
BUSINESS_NARRATIVE_PATH = ROOT / "assets" / "business_context_sfil.md"
OUT_BASE_DIR = ROOT / "data" / "intermediate" / "by_model"
SUMMARY_PATH = OUT_BASE_DIR / "generation_validation_summary.json"

DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODELS = [
    {"alias": "deepseek_r1", "model": "deepseek-ai/DeepSeek-R1"},
    {"alias": "mistral_small_24b", "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506"},
    {"alias": "gpt_oss_120b", "model": "openai/gpt-oss-120b"},
    {"alias": "qwen_7b", "model": "Qwen/Qwen2.5-7B-Instruct"},
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "model"


def parse_model_specs(raw: str) -> list[dict[str, str]]:
    raw = (raw or "").strip()
    if not raw:
        return [dict(item) for item in DEFAULT_MODELS]

    out: list[dict[str, str]] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "=" in token:
            alias, model = token.split("=", 1)
            alias = slugify(alias)
            model = model.strip()
        else:
            model = token
            alias = slugify(token)
        if not model:
            continue
        out.append({"alias": alias, "model": model})

    if not out:
        raise ValueError("--models produced an empty list.")
    return out


def model_output_paths(base_dir: Path, alias: str) -> dict[str, Path]:
    root = base_dir / alias
    return {
        "root": root,
        "raw_jsonl": root / "ttsql_candidates_raw.jsonl",
        "raw_csv": root / "ttsql_candidates_raw.csv",
        "validated_jsonl": root / "ttsql_candidates_validated.jsonl",
        "validated_csv": root / "ttsql_candidates_validated.csv",
        "clean_jsonl": root / "ttsql_candidates_clean.jsonl",
        "clean_csv": root / "ttsql_candidates_clean.csv",
        "report_json": root / "ttsql_candidates_validation_report.json",
    }


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def print_command(cmd: list[str]) -> None:
    rendered = " ".join(
        f'"{part}"' if (" " in part or "\t" in part) else part for part in cmd
    )
    print(f"[CMD] {rendered}")


def run_step(cmd: list[str], env: dict[str, str], dry_run: bool) -> int:
    print_command(cmd)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Script 03 (generation) then Script 04 (validation) for multiple DeepInfra models."
        )
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--schema-guide-path", type=Path, default=SCHEMA_GUIDE_PATH)
    parser.add_argument("--business-context-path", type=Path, default=BUSINESS_CONTEXT_PATH)
    parser.add_argument("--business-narrative-path", type=Path, default=BUSINESS_NARRATIVE_PATH)
    parser.add_argument("--out-base-dir", type=Path, default=OUT_BASE_DIR)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Comma-separated model specs. Format: alias=model_id or model_id. "
            "Example: deepseek=deepseek-ai/DeepSeek-R1,qwen=Qwen/Qwen2.5-7B-Instruct"
        ),
    )
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", type=str, default="DEEPINFRA_API_KEY")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--timeout-ms-validate", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--require-business-keyword", action="store_true")
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--require-declared-tables-match", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--max-rows-for-hash", type=int, default=5000)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_generation_cmd(
    args: argparse.Namespace, model_id: str, paths: dict[str, Path], model_index: int
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_03),
        "--db-path",
        str(args.db_path),
        "--schema-guide-path",
        str(args.schema_guide_path),
        "--business-context-path",
        str(args.business_context_path),
        "--business-narrative-path",
        str(args.business_narrative_path),
        "--out-jsonl",
        str(paths["raw_jsonl"]),
        "--out-csv",
        str(paths["raw_csv"]),
        "--num-questions",
        str(args.num_questions),
        "--batch-size",
        str(args.batch_size),
        "--max-calls",
        str(args.max_calls),
        "--temperature",
        str(args.temperature),
        "--timeout-s",
        str(args.timeout_s),
        "--base-url",
        str(args.base_url),
        "--model",
        model_id,
        "--api-key-env",
        args.api_key_env,
        "--seed",
        str(args.seed + model_index),
    ]
    if args.append:
        cmd.append("--append")
    if args.no_response_format:
        cmd.append("--no-response-format")
    if args.require_business_keyword:
        cmd.append("--require-business-keyword")
    return cmd


def build_validation_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_04),
        "--in-jsonl",
        str(paths["raw_jsonl"]),
        "--db-path",
        str(args.db_path),
        "--out-jsonl",
        str(paths["validated_jsonl"]),
        "--out-csv",
        str(paths["validated_csv"]),
        "--out-clean-jsonl",
        str(paths["clean_jsonl"]),
        "--out-clean-csv",
        str(paths["clean_csv"]),
        "--report-path",
        str(paths["report_json"]),
        "--timeout-ms",
        str(args.timeout_ms_validate),
        "--max-rows-for-hash",
        str(args.max_rows_for_hash),
    ]
    if args.require_non_empty:
        cmd.append("--require-non-empty")
    if args.require_declared_tables_match:
        cmd.append("--require-declared-tables-match")
    if args.keep_duplicates:
        cmd.append("--keep-duplicates")
    return cmd


def main() -> None:
    load_dotenv()
    args = parse_args()
    models = parse_model_specs(args.models)
    env = os.environ.copy()

    api_key = env.get(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError(
            f"Missing API key. Set {args.api_key_env} in your environment or .env file."
        )
    if not api_key and args.dry_run:
        print(f"[WARN] {args.api_key_env} is missing, but dry-run is enabled.")

    args.out_base_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "num_questions_per_model": args.num_questions,
        "models": [],
    }

    for idx, spec in enumerate(models):
        alias = spec["alias"]
        model_id = spec["model"]
        paths = model_output_paths(args.out_base_dir, alias)
        paths["root"].mkdir(parents=True, exist_ok=True)

        print(f"\n[MODEL] {alias} -> {model_id}")
        gen_cmd = build_generation_cmd(args, model_id, paths, idx)
        val_cmd = build_validation_cmd(args, paths)

        model_log: dict[str, Any] = {
            "alias": alias,
            "model": model_id,
            "paths": {k: str(v) for k, v in paths.items()},
            "generation_return_code": None,
            "validation_return_code": None,
            "generation_metrics": {},
            "status": "pending",
        }

        gen_rows_before = count_jsonl_rows(paths["raw_jsonl"])
        gen_t0 = time.perf_counter()
        gen_rc = run_step(gen_cmd, env=env, dry_run=args.dry_run)
        gen_elapsed_s = time.perf_counter() - gen_t0
        gen_rows_after = count_jsonl_rows(paths["raw_jsonl"])
        gen_rows_added = max(0, gen_rows_after - gen_rows_before)
        avg_latency_s = (
            (gen_elapsed_s / gen_rows_added) if gen_rows_added > 0 else None
        )
        avg_latency_ms = (avg_latency_s * 1000.0) if avg_latency_s is not None else None

        model_log["generation_metrics"] = {
            "generation_elapsed_seconds": round(gen_elapsed_s, 3),
            "raw_rows_before": gen_rows_before,
            "raw_rows_after": gen_rows_after,
            "raw_rows_added": gen_rows_added,
            "avg_generation_latency_seconds_per_query": (
                round(avg_latency_s, 3) if avg_latency_s is not None else None
            ),
            "avg_generation_latency_ms_per_query": (
                round(avg_latency_ms, 1) if avg_latency_ms is not None else None
            ),
        }
        if avg_latency_s is not None:
            print(
                "[METRIC] "
                f"{alias}: avg_generation_latency={avg_latency_s:.3f}s/query "
                f"({avg_latency_ms:.1f}ms/query) based_on={gen_rows_added} queries"
            )
        else:
            print(
                "[METRIC] "
                f"{alias}: avg_generation_latency unavailable "
                f"(raw_rows_added={gen_rows_added})"
            )

        model_log["generation_return_code"] = gen_rc
        if gen_rc != 0:
            model_log["status"] = "generation_failed"
            summary["models"].append(model_log)
            if args.continue_on_error:
                continue
            break

        val_rc = run_step(val_cmd, env=env, dry_run=args.dry_run)
        model_log["validation_return_code"] = val_rc
        if val_rc != 0:
            model_log["status"] = "validation_failed"
            summary["models"].append(model_log)
            if args.continue_on_error:
                continue
            break

        model_log["status"] = "ok"
        summary["models"].append(model_log)

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] Summary: {args.summary_path}")

    has_failures = any(item.get("status") != "ok" for item in summary["models"])
    if has_failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
