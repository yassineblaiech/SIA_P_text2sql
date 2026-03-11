import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_05 = ROOT / "scripts" / "05_judge_with_deepseek.py"
DB_PATH = ROOT / "data" / "raw" / "db.sqlite"
BUSINESS_CONTEXT_PATH = ROOT / "assets" / "schema_business_context.json"
BUSINESS_NARRATIVE_PATH = ROOT / "assets" / "business_context_sfil.md"
BY_MODEL_DIR = ROOT / "data" / "intermediate" / "by_model"
SUMMARY_PATH = BY_MODEL_DIR / "judge_all_summary.json"

DEFAULT_MODELS = ["deepseek_r1", "mistral_small_24b", "gpt_oss_120b", "qwen_7b"]
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_JUDGE_MODEL = "deepseek-chat"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_models(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return list(DEFAULT_MODELS)
    out = [token.strip() for token in raw.split(",") if token.strip()]
    if not out:
        raise ValueError("--models produced an empty list.")
    return out


def model_paths(base_dir: Path, alias: str) -> dict[str, Path]:
    root = base_dir / alias
    return {
        "root": root,
        "in_clean_jsonl": root / "ttsql_candidates_clean.jsonl",
        "out_judged_jsonl": root / "ttsql_candidates_judged.jsonl",
        "out_judged_csv": root / "ttsql_candidates_judged.csv",
        "out_pass_jsonl": root / "ttsql_candidates_judge_pass.jsonl",
        "out_pass_csv": root / "ttsql_candidates_judge_pass.csv",
        "out_report_json": root / "ttsql_candidates_judge_report.json",
    }


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
        description="Run Script 5 judge over all per-model clean datasets."
    )
    parser.add_argument("--by-model-dir", type=Path, default=BY_MODEL_DIR)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--business-context-path", type=Path, default=BUSINESS_CONTEXT_PATH)
    parser.add_argument("--business-narrative-path", type=Path, default=BUSINESS_NARRATIVE_PATH)
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.getenv("DEEPSEEK_MODEL_JUDGE", os.getenv("DEEPSEEK_MODEL", DEFAULT_JUDGE_MODEL)),
    )
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--pass-score-threshold", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=1.5)
    parser.add_argument("--schema-char-limit", type=int, default=12000)
    parser.add_argument("--business-char-limit", type=int, default=9000)
    parser.add_argument("--business-narrative-char-limit", type=int, default=16000)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_cmd(args: argparse.Namespace, paths: dict[str, Path], seed: int) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_05),
        "--in-jsonl",
        str(paths["in_clean_jsonl"]),
        "--db-path",
        str(args.db_path),
        "--business-context-path",
        str(args.business_context_path),
        "--business-narrative-path",
        str(args.business_narrative_path),
        "--out-jsonl",
        str(paths["out_judged_jsonl"]),
        "--out-csv",
        str(paths["out_judged_csv"]),
        "--out-pass-jsonl",
        str(paths["out_pass_jsonl"]),
        "--out-pass-csv",
        str(paths["out_pass_csv"]),
        "--report-path",
        str(paths["out_report_json"]),
        "--pass-score-threshold",
        str(args.pass_score_threshold),
        "--temperature",
        str(args.temperature),
        "--timeout-s",
        str(args.timeout_s),
        "--max-retries",
        str(args.max_retries),
        "--retry-backoff-s",
        str(args.retry_backoff_s),
        "--seed",
        str(seed),
        "--schema-char-limit",
        str(args.schema_char_limit),
        "--business-char-limit",
        str(args.business_char_limit),
        "--business-narrative-char-limit",
        str(args.business_narrative_char_limit),
        "--base-url",
        str(args.base_url),
        "--model",
        str(args.judge_model),
        "--api-key-env",
        str(args.api_key_env),
    ]
    if args.append:
        cmd.append("--append")
    if args.max_items > 0:
        cmd.extend(["--max-items", str(args.max_items)])
    return cmd


def try_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    load_dotenv()
    args = parse_args()
    models = parse_models(args.models)
    env = os.environ.copy()

    api_key = (env.get(args.api_key_env) or "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError(f"Missing API key. Set env var: {args.api_key_env}")
    if not api_key and args.dry_run:
        print(f"[WARN] {args.api_key_env} is missing, but dry-run is enabled.")

    args.by_model_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "models": [],
    }

    for idx, alias in enumerate(models):
        paths = model_paths(args.by_model_dir, alias)
        log: dict[str, Any] = {
            "alias": alias,
            "paths": {k: str(v) for k, v in paths.items()},
            "status": "pending",
            "return_code": None,
        }

        if not paths["in_clean_jsonl"].exists():
            log["status"] = "input_missing"
            summary["models"].append(log)
            print(f"[WARN] skip {alias}: missing {paths['in_clean_jsonl'].name}")
            continue

        cmd = build_cmd(args, paths, seed=42 + idx)
        rc = run_step(cmd, env=env, dry_run=args.dry_run)
        log["return_code"] = rc
        if rc != 0:
            log["status"] = "judge_failed"
            summary["models"].append(log)
            if args.continue_on_error:
                continue
            break

        log["status"] = "ok"
        report = try_read_json(paths["out_report_json"])
        if report:
            log["report"] = {
                "total_rows": report.get("total_rows"),
                "pass_rows": report.get("pass_rows"),
                "pass_rate": report.get("pass_rate"),
                "judge_errors": report.get("judge_errors"),
            }
        summary["models"].append(log)

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Summary: {args.summary_path}")

    has_failures = any(
        m.get("status") not in {"ok", "input_missing"} for m in summary["models"]
    )
    if has_failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
