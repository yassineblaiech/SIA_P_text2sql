import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "raw" / "db.sqlite"
IN_JSONL = ROOT / "data" / "intermediate" / "ttsql_candidates_raw.jsonl"
OUT_JSONL = ROOT / "data" / "intermediate" / "ttsql_candidates_validated.jsonl"
OUT_CSV = ROOT / "data" / "intermediate" / "ttsql_candidates_validated.csv"
OUT_CLEAN_JSONL = ROOT / "data" / "intermediate" / "ttsql_candidates_clean.jsonl"
OUT_CLEAN_CSV = ROOT / "data" / "intermediate" / "ttsql_candidates_clean.csv"
REPORT_PATH = ROOT / "data" / "intermediate" / "ttsql_candidates_validation_report.json"

BLOCKED_SQL_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "detach",
    "vacuum",
    "reindex",
    "replace",
    "pragma",
    "trigger",
}


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
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "question_fr",
                    "sql",
                    "difficulty",
                    "tables_used",
                    "tags",
                    "accept_for_training",
                    "reject_reasons",
                    "runtime_ok",
                    "runtime_error",
                    "runtime_timeout",
                    "execution_ms",
                    "result_row_count",
                    "result_col_count",
                    "result_hash_sha256",
                ]
            )
        return

    fieldnames = [
        "id",
        "question_fr",
        "sql",
        "difficulty",
        "tables_used",
        "tags",
        "accept_for_training",
        "reject_reasons",
        "runtime_ok",
        "runtime_error",
        "runtime_timeout",
        "execution_ms",
        "result_row_count",
        "result_col_count",
        "result_hash_sha256",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["tables_used"] = json.dumps(
                row.get("tables_used", []), ensure_ascii=False
            )
            payload["tags"] = json.dumps(row.get("tags", []), ensure_ascii=False)
            payload["reject_reasons"] = json.dumps(
                row.get("reject_reasons", []), ensure_ascii=False
            )
            writer.writerow(payload)


def known_tables(conn: sqlite3.Connection) -> list[str]:
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


def infer_tables_used(sql: str, tables: list[str]) -> list[str]:
    low = (sql or "").lower()
    out = []
    for table in tables:
        if re.search(rf"\b{re.escape(table.lower())}\b", low):
            out.append(table)
    return sorted(out)


def first_blocked_keyword(sql: str) -> str | None:
    low = (sql or "").lower()
    for kw in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return kw
    return None


def structure_and_safety_check(sql: str) -> tuple[bool, str, str]:
    raw = (sql or "").strip()
    if not raw:
        return False, "empty_sql", ""

    statements = [s.strip() for s in raw.split(";") if s.strip()]
    if len(statements) != 1:
        return False, "multi_statement", ""

    stmt = statements[0]
    head = stmt.lstrip().lower()
    if not (head.startswith("select ") or head.startswith("with ")):
        return False, "not_select_or_cte", stmt

    blocked = first_blocked_keyword(stmt)
    if blocked:
        return False, f"blocked_keyword:{blocked}", stmt

    return True, "", stmt


def execute_query_with_timeout(
    conn: sqlite3.Connection,
    sql: str,
    timeout_ms: int,
    max_rows_for_hash: int,
    fetch_chunk: int = 512,
) -> dict[str, Any]:
    start = time.perf_counter()
    deadline = start + (timeout_ms / 1000.0)
    timed_out = False

    def progress_handler() -> int:
        nonlocal timed_out
        if time.perf_counter() > deadline:
            timed_out = True
            return 1
        return 0

    result_row_count = 0
    result_col_count = 0
    hasher = hashlib.sha256()
    hashed_rows = 0

    conn.set_progress_handler(progress_handler, 5000)
    try:
        cur = conn.execute(sql)
        result_col_count = len(cur.description or [])

        while True:
            rows = cur.fetchmany(fetch_chunk)
            if not rows:
                break
            result_row_count += len(rows)

            for row in rows:
                if max_rows_for_hash < 0 or hashed_rows < max_rows_for_hash:
                    row_json = json.dumps(row, ensure_ascii=False, default=str)
                    hasher.update(row_json.encode("utf-8"))
                    hasher.update(b"\n")
                    hashed_rows += 1

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {
            "runtime_ok": True,
            "runtime_error": "",
            "runtime_timeout": False,
            "execution_ms": elapsed_ms,
            "result_row_count": result_row_count,
            "result_col_count": result_col_count,
            "result_hash_sha256": hasher.hexdigest(),
        }
    except sqlite3.Error as e:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        err = str(e)
        timeout_flag = timed_out or ("interrupted" in err.lower())
        return {
            "runtime_ok": False,
            "runtime_error": err,
            "runtime_timeout": timeout_flag,
            "execution_ms": elapsed_ms,
            "result_row_count": 0,
            "result_col_count": 0,
            "result_hash_sha256": "",
        }
    finally:
        conn.set_progress_handler(None, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated Text-to-SQL candidates by deterministic SQLite execution."
    )
    parser.add_argument("--in-jsonl", type=Path, default=IN_JSONL)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-clean-jsonl", type=Path, default=OUT_CLEAN_JSONL)
    parser.add_argument("--out-clean-csv", type=Path, default=OUT_CLEAN_CSV)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--timeout-ms", type=int, default=8000)
    parser.add_argument("--max-rows-for-hash", type=int, default=5000)
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--require-declared-tables-match", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.in_jsonl)
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]

    conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON;")
    tables = known_tables(conn)

    validated_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    reason_counts = Counter()
    seen_q = set()
    seen_sql = set()

    for idx, row in enumerate(rows, start=1):
        question = normalize_spaces(str(row.get("question_fr", "")))
        sql = (row.get("sql") or "").strip()
        declared_tables = sorted(set(row.get("tables_used") or []))
        inferred_tables = infer_tables_used(sql, tables)

        validated = dict(row)
        validated["question_fr"] = question
        validated["sql"] = sql
        validated["inferred_tables_used"] = inferred_tables
        validated["structure_ok"] = False
        validated["safety_ok"] = False
        validated["runtime_ok"] = False
        validated["runtime_error"] = ""
        validated["runtime_timeout"] = False
        validated["execution_ms"] = 0
        validated["result_row_count"] = 0
        validated["result_col_count"] = 0
        validated["result_hash_sha256"] = ""
        validated["reject_reasons"] = []
        validated["accept_for_training"] = False
        validated["validated_at_utc"] = now_utc()

        ok, struct_reason, stmt = structure_and_safety_check(sql)
        if not ok:
            validated["reject_reasons"].append(struct_reason)
            reason_counts[struct_reason] += 1
            validated_rows.append(validated)
            continue

        validated["structure_ok"] = True
        validated["safety_ok"] = True

        if args.require_declared_tables_match:
            if declared_tables != inferred_tables:
                reason = "declared_tables_mismatch"
                validated["reject_reasons"].append(reason)
                reason_counts[reason] += 1

        run = execute_query_with_timeout(
            conn=conn,
            sql=stmt,
            timeout_ms=max(1, args.timeout_ms),
            max_rows_for_hash=args.max_rows_for_hash,
        )
        validated.update(run)

        if not validated["runtime_ok"]:
            reason = "runtime_timeout" if validated["runtime_timeout"] else "runtime_error"
            validated["reject_reasons"].append(reason)
            reason_counts[reason] += 1

        if args.require_non_empty and validated["runtime_ok"] and validated["result_row_count"] == 0:
            reason = "empty_result"
            validated["reject_reasons"].append(reason)
            reason_counts[reason] += 1

        q_norm = normalize_spaces(question.lower())
        sql_norm = normalize_spaces(sql.lower())
        if not args.keep_duplicates:
            if q_norm in seen_q:
                reason = "duplicate_question"
                validated["reject_reasons"].append(reason)
                reason_counts[reason] += 1
            if sql_norm in seen_sql:
                reason = "duplicate_sql"
                validated["reject_reasons"].append(reason)
                reason_counts[reason] += 1

        if not validated["reject_reasons"]:
            validated["accept_for_training"] = True
            clean_rows.append(validated)

        validated_rows.append(validated)
        seen_q.add(q_norm)
        seen_sql.add(sql_norm)

        if idx % 25 == 0:
            print(f"[INFO] validated {idx}/{len(rows)}")

    conn.close()

    write_jsonl(args.out_jsonl, validated_rows)
    write_csv(args.out_csv, validated_rows)
    write_jsonl(args.out_clean_jsonl, clean_rows)
    write_csv(args.out_clean_csv, clean_rows)

    total = len(validated_rows)
    accepted = len(clean_rows)
    rejected = total - accepted
    by_diff_total = Counter((r.get("difficulty") or "").lower() for r in validated_rows)
    by_diff_accepted = Counter((r.get("difficulty") or "").lower() for r in clean_rows)

    report = {
        "generated_at_utc": now_utc(),
        "input_path": str(args.in_jsonl),
        "db_path": str(args.db_path),
        "validated_rows": total,
        "accepted_rows": accepted,
        "rejected_rows": rejected,
        "acceptance_rate": (accepted / total) if total else 0.0,
        "require_non_empty": args.require_non_empty,
        "require_declared_tables_match": args.require_declared_tables_match,
        "keep_duplicates": args.keep_duplicates,
        "timeout_ms": args.timeout_ms,
        "max_rows_for_hash": args.max_rows_for_hash,
        "counts_by_reject_reason": dict(reason_counts),
        "difficulty_total": dict(by_diff_total),
        "difficulty_accepted": dict(by_diff_accepted),
        "runtime_ok_count": sum(1 for r in validated_rows if r.get("runtime_ok")),
        "runtime_error_count": sum(1 for r in validated_rows if not r.get("runtime_ok")),
        "runtime_timeout_count": sum(1 for r in validated_rows if r.get("runtime_timeout")),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Validated JSONL: {args.out_jsonl} ({total} rows)")
    print(f"[OK] Validated CSV:   {args.out_csv} ({total} rows)")
    print(f"[OK] Clean JSONL:     {args.out_clean_jsonl} ({accepted} rows)")
    print(f"[OK] Clean CSV:       {args.out_clean_csv} ({accepted} rows)")
    print(f"[OK] Report:          {args.report_path}")
    print(
        f"[SUMMARY] accepted={accepted} rejected={rejected} "
        f"acceptance_rate={report['acceptance_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
