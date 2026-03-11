import argparse
import csv
import json
import os
import random
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "raw" / "db.sqlite"
IN_JSONL = ROOT / "data" / "intermediate" / "ttsql_candidates_clean.jsonl"
BUSINESS_CONTEXT_PATH = ROOT / "assets" / "schema_business_context.json"
BUSINESS_NARRATIVE_PATH = ROOT / "assets" / "business_context_sfil.md"
OUT_JSONL = ROOT / "data" / "intermediate" / "ttsql_candidates_judged.jsonl"
OUT_CSV = ROOT / "data" / "intermediate" / "ttsql_candidates_judged.csv"
OUT_PASS_JSONL = ROOT / "data" / "intermediate" / "ttsql_candidates_judge_pass.jsonl"
OUT_PASS_CSV = ROOT / "data" / "intermediate" / "ttsql_candidates_judge_pass.csv"
REPORT_PATH = ROOT / "data" / "intermediate" / "ttsql_candidates_judge_report.json"

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON line {line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question_fr",
        "sql",
        "difficulty",
        "tables_used",
        "runtime_ok",
        "runtime_error",
        "result_row_count",
        "judge_score",
        "judge_verdict",
        "judge_reason",
        "judge_pass",
        "judge_fix_sql",
        "judge_dimension_scores",
        "judge_flags",
        "judge_model",
        "judged_at_utc",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["tables_used"] = json.dumps(
                row.get("tables_used", []), ensure_ascii=False
            )
            payload["judge_dimension_scores"] = json.dumps(
                row.get("judge_dimension_scores", {}), ensure_ascii=False
            )
            payload["judge_flags"] = json.dumps(
                row.get("judge_flags", []), ensure_ascii=False
            )
            writer.writerow(payload)


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
          AND name NOT LIKE 'stg_%'
        ORDER BY name;
        """
    ).fetchall()
    return [r[0] for r in rows]


def build_schema_snapshot(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    lines = []
    for table in table_names(conn):
        lines.append(f"TABLE {table}")
        cols = conn.execute(f'PRAGMA table_info("{table}");').fetchall()
        fk_rows = conn.execute(f'PRAGMA foreign_key_list("{table}");').fetchall()
        fk_map = {r[3]: (r[2], r[4]) for r in fk_rows}
        for c in cols:
            name = c[1]
            col_type = c[2] or "TEXT"
            not_null = bool(c[3])
            is_pk = bool(c[5])
            bits = [name, col_type]
            if is_pk:
                bits.append("PK")
            if not_null:
                bits.append("NOT_NULL")
            if name in fk_map:
                to_t, to_c = fk_map[name]
                bits.append(f"FK->{to_t}.{to_c}")
            lines.append("  - " + " | ".join(bits))
    conn.close()
    return "\n".join(lines)


def load_business_context(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def compact_business_notes(context: dict[str, Any], max_chars: int) -> str:
    if not context:
        return ""
    lines: list[str] = []
    global_notes = context.get("global_notes", [])
    if isinstance(global_notes, list):
        lines.append("Global notes:")
        for note in global_notes:
            lines.append(f"- {note}")

    tables = context.get("tables", {})
    if isinstance(tables, dict):
        for table, payload in tables.items():
            lines.append(f"\nTable {table}:")
            if isinstance(payload, dict):
                if "role" in payload:
                    lines.append(f"role: {payload['role']}")
                rels = payload.get("relationships", [])
                if isinstance(rels, list):
                    for rel in rels:
                        lines.append(f"relationship: {rel}")
                hints = payload.get("query_hints", [])
                if isinstance(hints, list):
                    for hint in hints:
                        lines.append(f"hint: {hint}")
                if payload.get("empty_table_is_normal") is True:
                    lines.append("hint: this table may be empty and that can be normal")

    return "\n".join(lines)[:max_chars]


def load_text_excerpt(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return text[:max_chars]


def extract_json_object(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge response")
    return raw[start : end + 1]


def call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_judge_payload(raw_text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_object(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Judge payload is not an object")

    score = payload.get("score")
    verdict = str(payload.get("verdict", "needs_fix")).strip().lower()
    reason = normalize_spaces(str(payload.get("reason", "")))
    fix_sql = str(payload.get("fix_sql", "")).strip()
    dim = payload.get("dimension_scores", {})
    flags = payload.get("flags", [])

    if not isinstance(score, (int, float)):
        score = 0
    score = max(0.0, min(5.0, float(score)))

    if verdict not in {"pass", "fail", "needs_fix"}:
        verdict = "needs_fix"

    if not isinstance(dim, dict):
        dim = {}
    norm_dim = {}
    for k, v in dim.items():
        if isinstance(v, (int, float)):
            norm_dim[str(k)] = max(0.0, min(5.0, float(v)))

    if not isinstance(flags, list):
        flags = []
    norm_flags = [normalize_spaces(str(x).lower()) for x in flags if str(x).strip()]

    return {
        "score": score,
        "verdict": verdict,
        "reason": reason,
        "fix_sql": fix_sql,
        "dimension_scores": norm_dim,
        "flags": norm_flags,
    }


def build_messages(
    row: dict[str, Any],
    schema_snapshot: str,
    business_notes: str,
    business_narrative: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a strict Text-to-SQL judge for SQLite.\n"
        "Evaluate if SQL matches the French question intent and schema constraints.\n"
        "Return JSON only with keys:\n"
        "{\n"
        '  "score": 0..5,\n'
        '  "verdict": "pass|needs_fix|fail",\n'
        '  "reason": "short explanation",\n'
        '  "dimension_scores": {\n'
        '      "intent_alignment": 0..5,\n'
        '      "schema_correctness": 0..5,\n'
        '      "query_quality": 0..5\n'
        "  },\n"
        '  "flags": ["optional_flags"],\n'
        '  "fix_sql": "improved SQL if needed else empty string"\n'
        "}\n"
        "Judge rubric:\n"
        "- intent_alignment: Does SQL answer the question asked?\n"
        "- schema_correctness: valid columns/tables/joins and SQLite compatibility.\n"
        "- query_quality: clarity, unnecessary complexity, robustness.\n"
        "- business_realism: question wording should sound like a financial analyst, not a technical DB user.\n"
        "Penalize heavily if question mentions table/column/join/schema/SQL internals.\n"
        "Use 'pass' only when SQL is clearly good enough for training."
    )

    user_prompt = (
        "Business narrative (SFIL context):\n"
        f"{business_narrative or '-'}\n\n"
        "Schema snapshot:\n"
        f"{schema_snapshot}\n\n"
        "Business notes:\n"
        f"{business_notes or '-'}\n\n"
        "Candidate:\n"
        f"id: {row.get('id')}\n"
        f"question_fr: {row.get('question_fr')}\n"
        f"sql: {row.get('sql')}\n"
        f"difficulty: {row.get('difficulty')}\n"
        f"tables_used: {json.dumps(row.get('tables_used', []), ensure_ascii=False)}\n"
        f"runtime_ok: {row.get('runtime_ok')}\n"
        f"runtime_error: {row.get('runtime_error')}\n"
        f"result_row_count: {row.get('result_row_count')}\n"
        "Return JSON only."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script 5: LLM-as-judge scoring for Text-to-SQL candidates."
    )
    parser.add_argument("--in-jsonl", type=Path, default=IN_JSONL)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--business-context-path", type=Path, default=BUSINESS_CONTEXT_PATH)
    parser.add_argument("--business-narrative-path", type=Path, default=BUSINESS_NARRATIVE_PATH)
    parser.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-pass-jsonl", type=Path, default=OUT_PASS_JSONL)
    parser.add_argument("--out-pass-csv", type=Path, default=OUT_PASS_CSV)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--pass-score-threshold", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema-char-limit", type=int, default=12000)
    parser.add_argument("--business-char-limit", type=int, default=9000)
    parser.add_argument("--business-narrative-char-limit", type=int, default=16000)
    parser.add_argument("--base-url", type=str, default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", type=str, default=os.getenv("DEEPSEEK_MODEL_JUDGE", os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    rng = random.Random(args.seed)

    rows = load_jsonl(args.in_jsonl)
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]

    schema_snapshot = build_schema_snapshot(args.db_path)[: args.schema_char_limit]
    business_notes = compact_business_notes(
        load_business_context(args.business_context_path),
        max_chars=args.business_char_limit,
    )
    business_narrative = load_text_excerpt(
        args.business_narrative_path, max_chars=args.business_narrative_char_limit
    )

    if args.dry_run:
        if not rows:
            print("[WARN] No input rows for dry-run.")
            return
        preview_row = rows[0]
        preview = build_messages(
            preview_row, schema_snapshot, business_notes, business_narrative
        )
        print(json.dumps({"messages": preview}, ensure_ascii=False, indent=2)[:7000])
        print("[OK] Dry run prompt preview ready.")
        return

    api_key = (os.getenv(args.api_key_env) or "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    existing_judged: dict[str, dict[str, Any]] = {}
    if args.append and args.out_jsonl.exists():
        for r in load_jsonl(args.out_jsonl):
            rid = str(r.get("id", "")).strip()
            if rid:
                existing_judged[rid] = r

    judged_rows: list[dict[str, Any]] = []
    verdict_counts = Counter()
    score_buckets = Counter()
    errors_count = 0

    for idx, row in enumerate(rows, start=1):
        rid = str(row.get("id", "")).strip()
        if rid and rid in existing_judged:
            judged_rows.append(existing_judged[rid])
            continue

        messages = build_messages(
            row, schema_snapshot, business_notes, business_narrative
        )
        raw_response = ""
        parsed: dict[str, Any] | None = None
        call_error = ""

        for attempt in range(1, args.max_retries + 1):
            try:
                raw_response = call_chat_completions(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                )
                parsed = parse_judge_payload(raw_response)
                break
            except Exception as e:
                call_error = str(e)
                if attempt >= args.max_retries:
                    break
                # jitter avoids synchronized retries
                delay = args.retry_backoff_s * attempt + rng.random() * 0.25
                time.sleep(delay)

        out = dict(row)
        out["judge_model"] = args.model
        out["judged_at_utc"] = now_utc()

        if parsed is None:
            errors_count += 1
            out["judge_score"] = 0.0
            out["judge_verdict"] = "fail"
            out["judge_reason"] = normalize_spaces(
                f"judge_error: {call_error}" if call_error else "judge_error: unknown"
            )
            out["judge_fix_sql"] = ""
            out["judge_dimension_scores"] = {}
            out["judge_flags"] = ["judge_error"]
            out["judge_pass"] = False
            out["judge_raw_response"] = raw_response
        else:
            score = float(parsed["score"])
            verdict = parsed["verdict"]
            reason = parsed["reason"]
            fix_sql = parsed["fix_sql"]
            dim = parsed["dimension_scores"]
            flags = parsed["flags"]

            pass_by_score = score >= args.pass_score_threshold
            pass_by_verdict = verdict == "pass"
            judge_pass = pass_by_score and pass_by_verdict

            out["judge_score"] = score
            out["judge_verdict"] = verdict
            out["judge_reason"] = reason
            out["judge_fix_sql"] = fix_sql
            out["judge_dimension_scores"] = dim
            out["judge_flags"] = flags
            out["judge_pass"] = judge_pass
            out["judge_raw_response"] = raw_response

            verdict_counts[verdict] += 1
            score_buckets[f"{int(score)}"] += 1

        judged_rows.append(out)

        if idx % 20 == 0 or idx == len(rows):
            print(f"[INFO] judged {idx}/{len(rows)}")

    # Keep original input order in output
    by_id = {str(r.get("id", "")): r for r in judged_rows if str(r.get("id", "")).strip()}
    ordered_judged = []
    for r in rows:
        rid = str(r.get("id", "")).strip()
        ordered_judged.append(by_id.get(rid, r))

    pass_rows = [r for r in ordered_judged if r.get("judge_pass") is True]

    write_jsonl(args.out_jsonl, ordered_judged)
    write_csv(args.out_csv, ordered_judged)
    write_jsonl(args.out_pass_jsonl, pass_rows)
    write_csv(args.out_pass_csv, pass_rows)

    report = {
        "generated_at_utc": now_utc(),
        "input_path": str(args.in_jsonl),
        "db_path": str(args.db_path),
        "judge_model": args.model,
        "base_url": args.base_url,
        "total_rows": len(ordered_judged),
        "pass_rows": len(pass_rows),
        "pass_rate": (len(pass_rows) / len(ordered_judged)) if ordered_judged else 0.0,
        "pass_score_threshold": args.pass_score_threshold,
        "verdict_counts": dict(verdict_counts),
        "score_bucket_counts": dict(score_buckets),
        "judge_errors": errors_count,
        "difficulty_total": dict(
            Counter((r.get("difficulty") or "").lower() for r in ordered_judged)
        ),
        "difficulty_pass": dict(
            Counter((r.get("difficulty") or "").lower() for r in pass_rows)
        ),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Judged JSONL: {args.out_jsonl} ({len(ordered_judged)} rows)")
    print(f"[OK] Judged CSV:   {args.out_csv} ({len(ordered_judged)} rows)")
    print(f"[OK] Pass JSONL:   {args.out_pass_jsonl} ({len(pass_rows)} rows)")
    print(f"[OK] Pass CSV:     {args.out_pass_csv} ({len(pass_rows)} rows)")
    print(f"[OK] Report:       {args.report_path}")
    print(f"[SUMMARY] pass={len(pass_rows)} total={len(ordered_judged)}")


if __name__ == "__main__":
    main()
