import argparse
import csv
import http.client
import json
import os
import random
import re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "raw" / "db.sqlite"
SCHEMA_GUIDE_PATH = ROOT / "assets" / "schema_guide.yaml"
BUSINESS_CONTEXT_PATH = ROOT / "assets" / "schema_business_context.json"
BUSINESS_NARRATIVE_PATH = ROOT / "assets" / "business_context_sfil.md"
OUT_JSONL_PATH = ROOT / "data" / "intermediate" / "ttsql_candidates_raw.jsonl"
OUT_CSV_PATH = ROOT / "data" / "intermediate" / "ttsql_candidates_raw.csv"

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-reasoner"

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

TECHNICAL_QUESTION_PATTERNS = [
    r"\btable\b",
    r"\bcolonne\b",
    r"\bschéma\b",
    r"\bschema\b",
    r"\bbase de donn[ée]es\b",
    r"\bbdd\b",
    r"\bjointure\b",
    r"\bjoin\b",
    r"\bsql\b",
    r"\bfk\b",
    r"\bpk\b",
]

BUSINESS_HINT_PATTERNS = [
    r"\bcommune\b",
    r"\bmunicipalit[ée]\b",
    r"\bbudget\b",
    r"\bd[ée]pense",
    r"\brecette",
    r"\brevenu",
    r"\binvestissement",
    r"\bdette\b",
    r"\bann[ée]e\b",
    r"\bindicateur",
    r"\bfinanci",
    r"\bextraction\b",
    r"\br[èe]gle\b",
    r"\bvalidation\b",
    r"\bdocument\b",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_difficulty_profile(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    allowed = {"easy", "medium", "hard"}
    for chunk in raw.split(","):
        part = chunk.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid difficulty profile chunk: {part}")
        key, value = part.split(":", 1)
        key = key.strip().lower()
        if key not in allowed:
            raise ValueError(f"Unknown difficulty level: {key}")
        out[key] = float(value.strip())

    if set(out.keys()) != allowed:
        raise ValueError("Difficulty profile must define easy,medium,hard.")

    total = sum(out.values())
    if total <= 0:
        raise ValueError("Difficulty profile total weight must be > 0.")
    return {k: v / total for k, v in out.items()}


def allocate_difficulties(
    n: int, profile: dict[str, float], rng: random.Random
) -> dict[str, int]:
    targets = {k: n * v for k, v in profile.items()}
    counts = {k: int(targets[k]) for k in profile}
    remaining = n - sum(counts.values())

    if remaining > 0:
        order = sorted(
            targets.keys(),
            key=lambda k: (targets[k] - counts[k], rng.random()),
            reverse=True,
        )
        for key in order[:remaining]:
            counts[key] += 1
    return counts


def extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return text[start : end + 1]


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"

    fieldnames = [
        "id",
        "question_fr",
        "sql",
        "difficulty",
        "tags",
        "tables_used",
        "valid_sql",
        "validation_error",
        "generation_model",
        "generated_at_utc",
        "batch_index",
    ]

    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["tags"] = json.dumps(row.get("tags", []), ensure_ascii=False)
            payload["tables_used"] = json.dumps(
                row.get("tables_used", []), ensure_ascii=False
            )
            writer.writerow(payload)


def list_tables(conn: sqlite3.Connection) -> list[str]:
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


def parse_check_in_values(table_sql: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not table_sql:
        return out

    sql = table_sql.replace("\n", " ")
    pattern = re.compile(
        r"""CHECK\s*\(\s*"?([A-Za-z_][A-Za-z0-9_]*)"?\s+IN\s*\(([^)]*)\)\s*\)""",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql):
        col = match.group(1)
        body = match.group(2)
        values = []
        for raw in body.split(","):
            token = raw.strip()
            if token.startswith("'") and token.endswith("'") and len(token) >= 2:
                values.append(token[1:-1].replace("''", "'"))
        if values:
            out[col] = values
    return out


def build_schema_snapshot(db_path: Path) -> tuple[list[str], str]:
    conn = sqlite3.connect(db_path)
    tables = list_tables(conn)
    lines = []
    for table in tables:
        lines.append(f"TABLE {table}")
        cols = conn.execute(f'PRAGMA table_info("{table}");').fetchall()
        fk_rows = conn.execute(f'PRAGMA foreign_key_list("{table}");').fetchall()
        fk_map = {r[3]: (r[2], r[4]) for r in fk_rows}
        table_sql_row = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type='table' AND name=?;
            """,
            (table,),
        ).fetchone()
        table_sql = table_sql_row[0] if table_sql_row else ""
        check_values = parse_check_in_values(table_sql)
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
            if name in check_values:
                bits.append("ENUM{" + "|".join(check_values[name]) + "}")
            lines.append("  - " + " | ".join(bits))
    conn.close()
    return tables, "\n".join(lines)


def load_business_context(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return data


def compact_business_notes(context: dict[str, Any], max_chars: int = 12000) -> str:
    if not context:
        return ""

    lines = []
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

    text = "\n".join(lines).strip()
    return text[:max_chars]


def load_schema_guide_excerpt(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[:max_chars]


def load_text_excerpt(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return text[:max_chars]


def question_is_too_technical(question: str) -> bool:
    q = normalize_spaces(question).lower()
    if not q:
        return True
    return any(re.search(p, q) for p in TECHNICAL_QUESTION_PATTERNS)


def question_is_business_like(question: str) -> bool:
    q = normalize_spaces(question).lower()
    return any(re.search(p, q) for p in BUSINESS_HINT_PATTERNS)


def build_messages(
    schema_snapshot: str,
    schema_guide_excerpt: str,
    business_notes: str,
    business_narrative: str,
    batch_size: int,
    difficulty_counts: dict[str, int],
    existing_questions: list[str],
) -> list[dict[str, str]]:
    existing_sample = existing_questions[-50:]
    avoid_block = "\n".join(f"- {q}" for q in existing_sample) if existing_sample else "-"

    system_prompt = (
        "You generate high-quality Text-to-SQL training examples.\n"
        "Output must be valid JSON only.\n"
        "Target SQL dialect: SQLite.\n"
        "Hard constraints:\n"
        "1) Generate only read-only SQL (SELECT or WITH ... SELECT).\n"
        "2) Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, PRAGMA, ATTACH, DETACH.\n"
        "3) Use only existing tables/columns from provided schema.\n"
        "4) Questions must be in French.\n"
        "5) Questions must sound like business users (financial analysts), not technical users.\n"
        "6) Never mention SQL internals in question text (table, colonne, join, FK, schema, base de données).\n"
        "7) Favor realistic public-finance analytics questions.\n"
        "8) Ensure variety in joins, filters, aggregations, ordering, and grouping.\n"
        "9) Avoid duplicates and near-duplicates.\n"
        "JSON schema:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "question_fr": "...",\n'
        '      "sql": "...",\n'
        '      "difficulty": "easy|medium|hard",\n'
        '      "tags": ["...", "..."],\n'
        '      "tables_used": ["table1", "table2"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Nombre attendu: {batch_size}\n"
        f"Répartition difficulté demandée: easy={difficulty_counts['easy']}, "
        f"medium={difficulty_counts['medium']}, hard={difficulty_counts['hard']}\n\n"
        "Contexte métier prioritaire (SFIL):\n"
        f"{business_narrative or '-'}\n\n"
        "Schema snapshot:\n"
        f"{schema_snapshot}\n\n"
        "Business notes (manager):\n"
        f"{business_notes or '-'}\n\n"
        "Schema guide excerpt:\n"
        f"{schema_guide_excerpt or '-'}\n\n"
        "Questions déjà générées à éviter (échantillon):\n"
        f"{avoid_block}\n\n"
        "Important:\n"
        "- Les questions doivent être formulées comme des analystes financiers métier.\n"
        "- Interdit dans la question: référence explicite à table/colonne/join/SQL/schéma.\n"
        "- Mauvais style: 'joindre la table states'.\n"
        "- Bon style: 'Quel est le taux de documents en erreur par année ?'.\n"
        "- Utilise des jointures cohérentes avec les FK.\n"
        "- Évite de dépendre de tables potentiellement vides (ex: resultfeedbacks) sauf si explicite.\n"
        "- Les tags doivent être en snake_case.\n"
        "- tables_used doit refléter réellement le SQL.\n"
        "- Vise des analyses finances locales: tendance, comparaison, agrégat, anomalie, structure budgétaire.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_s: int,
    use_response_format: bool,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_items(model_text: str) -> list[dict[str, Any]]:
    raw = extract_json_object(model_text)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Model payload is not an object.")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Payload missing 'items' list.")

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = normalize_spaces(str(item.get("question_fr", "")))
        sql = str(item.get("sql", "")).strip()
        difficulty = str(item.get("difficulty", "medium")).lower().strip()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"
        tags = item.get("tags", [])
        tables_used = item.get("tables_used", [])

        if not isinstance(tags, list):
            tags = []
        if not isinstance(tables_used, list):
            tables_used = []
        tags = [normalize_spaces(str(t)).lower().replace(" ", "_") for t in tags if str(t).strip()]
        tables_used = [normalize_spaces(str(t)) for t in tables_used if str(t).strip()]

        if not question or not sql:
            continue
        out.append(
            {
                "question_fr": question,
                "sql": sql,
                "difficulty": difficulty,
                "tags": tags,
                "tables_used": tables_used,
            }
        )
    return out


def first_blocked_keyword(sql: str) -> str | None:
    low = sql.lower()
    for kw in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return kw
    return None


def validate_sql(sql: str, conn: sqlite3.Connection) -> tuple[bool, str]:
    raw = sql.strip()
    if not raw:
        return False, "empty_sql"

    statements = [s.strip() for s in raw.split(";") if s.strip()]
    if len(statements) != 1:
        return False, "multi_statement"
    stmt = statements[0]
    head = stmt.lstrip().lower()
    if not (head.startswith("select ") or head.startswith("with ")):
        return False, "not_select_or_cte"

    blocked = first_blocked_keyword(stmt)
    if blocked:
        return False, f"blocked_keyword:{blocked}"

    try:
        conn.execute(f"EXPLAIN QUERY PLAN {stmt}")
    except sqlite3.Error as e:
        return False, f"sqlite_error:{e}"

    return True, ""


def infer_tables_used(sql: str, known_tables: list[str]) -> list[str]:
    found = []
    low = sql.lower()
    for table in known_tables:
        if re.search(rf"\b{re.escape(table.lower())}\b", low):
            found.append(table)
    return found


def next_id(existing_rows: list[dict[str, Any]]) -> int:
    mx = 0
    for row in existing_rows:
        ident = str(row.get("id", ""))
        m = re.search(r"(\d+)$", ident)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate French Text-to-SQL candidates with DeepSeek."
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--schema-guide-path", type=Path, default=SCHEMA_GUIDE_PATH)
    parser.add_argument("--business-context-path", type=Path, default=BUSINESS_CONTEXT_PATH)
    parser.add_argument(
        "--business-narrative-path",
        type=Path,
        default=BUSINESS_NARRATIVE_PATH,
    )
    parser.add_argument("--out-jsonl", type=Path, default=OUT_JSONL_PATH)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV_PATH)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-calls", type=int, default=30)
    parser.add_argument("--difficulty-profile", type=str, default="easy:0.4,medium:0.4,hard:0.2")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", type=str, default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Disable OpenAI response_format=json_object for providers/models that do not support it.",
    )
    parser.add_argument("--schema-guide-char-limit", type=int, default=14000)
    parser.add_argument("--business-notes-char-limit", type=int, default=12000)
    parser.add_argument("--business-narrative-char-limit", type=int, default=16000)
    parser.add_argument("--require-business-keyword", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.num_questions <= 0:
        raise ValueError("--num-questions must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    rng = random.Random(args.seed)
    difficulty_profile = parse_difficulty_profile(args.difficulty_profile)

    known_tables, schema_snapshot = build_schema_snapshot(args.db_path)
    schema_guide_excerpt = load_schema_guide_excerpt(
        args.schema_guide_path, args.schema_guide_char_limit
    )
    business_context = load_business_context(args.business_context_path)
    business_notes = compact_business_notes(
        business_context, max_chars=args.business_notes_char_limit
    )
    business_narrative = load_text_excerpt(
        args.business_narrative_path, max_chars=args.business_narrative_char_limit
    )

    if args.dry_run:
        difficulty_counts = allocate_difficulties(args.batch_size, difficulty_profile, rng)
        preview = build_messages(
            schema_snapshot=schema_snapshot,
            schema_guide_excerpt=schema_guide_excerpt,
            business_notes=business_notes,
            business_narrative=business_narrative,
            batch_size=args.batch_size,
            difficulty_counts=difficulty_counts,
            existing_questions=[],
        )
        print("[INFO] Dry run mode: prompt preview generated.")
        print(json.dumps({"messages": preview}, ensure_ascii=False, indent=2)[:6000])
        return

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set env var {args.api_key_env} (for example in .env)."
        )

    existing_rows = load_existing_records(args.out_jsonl) if args.append else []
    initial_count = len(existing_rows)
    target_new = max(0, args.num_questions - initial_count)

    if target_new == 0:
        print(
            f"[INFO] Nothing to generate: existing={initial_count} "
            f"already >= requested={args.num_questions}"
        )
        return
    existing_qset = {normalize_spaces(str(r.get("question_fr", "")).lower()) for r in existing_rows}
    existing_sql_set = {normalize_spaces(str(r.get("sql", "")).lower()) for r in existing_rows}

    conn = sqlite3.connect(args.db_path)

    generated: list[dict[str, Any]] = []
    call_idx = 0
    next_numeric_id = next_id(existing_rows)
    filtered_too_technical = 0
    filtered_not_business_like = 0
    filtered_duplicate_question = 0
    filtered_duplicate_sql = 0
    filtered_invalid_sql = 0

    while len(generated) < target_new and call_idx < args.max_calls:
        call_idx += 1
        need = target_new - len(generated)
        this_batch = min(args.batch_size, need)
        difficulty_counts = allocate_difficulties(this_batch, difficulty_profile, rng)

        messages = build_messages(
            schema_snapshot=schema_snapshot,
            schema_guide_excerpt=schema_guide_excerpt,
            business_notes=business_notes,
            business_narrative=business_narrative,
            batch_size=this_batch,
            difficulty_counts=difficulty_counts,
            existing_questions=[r["question_fr"] for r in existing_rows + generated],
        )

        model_text = ""
        for attempt in range(1, 6):
            try:
                model_text = call_chat_completions(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                    use_response_format=not args.no_response_format,
                )
                break
            except urllib.error.HTTPError as e:
                # Invalid credentials should fail fast.
                if e.code == 401:
                    raise RuntimeError(
                        f"API authentication failed (401). Check {args.api_key_env}."
                    ) from e
                if attempt == 5:
                    raise RuntimeError(f"API call failed after retries: {e}") from e
                time.sleep(1.5 * attempt)
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.IncompleteRead,
                ConnectionResetError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as e:
                if attempt == 5:
                    raise RuntimeError(f"API call failed after retries: {e}") from e
                time.sleep(1.5 * attempt)

        try:
            items = parse_items(model_text)
        except Exception as e:
            print(f"[WARN] Batch {call_idx}: invalid JSON payload ({e}).")
            continue

        accepted = 0
        batch_duplicate_question = 0
        batch_duplicate_sql = 0
        batch_too_technical = 0
        batch_not_business_like = 0
        batch_invalid_sql = 0
        for item in items:
            q_norm = normalize_spaces(item["question_fr"].lower())
            sql_norm = normalize_spaces(item["sql"].lower())
            if q_norm in existing_qset:
                filtered_duplicate_question += 1
                batch_duplicate_question += 1
                continue
            if sql_norm in existing_sql_set:
                filtered_duplicate_sql += 1
                batch_duplicate_sql += 1
                continue

            if question_is_too_technical(item["question_fr"]):
                filtered_too_technical += 1
                batch_too_technical += 1
                continue
            if args.require_business_keyword and not question_is_business_like(
                item["question_fr"]
            ):
                filtered_not_business_like += 1
                batch_not_business_like += 1
                continue

            valid_sql, validation_error = validate_sql(item["sql"], conn)
            if not valid_sql:
                filtered_invalid_sql += 1
                batch_invalid_sql += 1
                continue

            if not item["tables_used"]:
                item["tables_used"] = infer_tables_used(item["sql"], known_tables)

            record = {
                "id": f"cand_{next_numeric_id:06d}",
                "question_fr": item["question_fr"],
                "sql": item["sql"].strip(),
                "difficulty": item["difficulty"],
                "tags": item["tags"],
                "tables_used": item["tables_used"],
                "valid_sql": valid_sql,
                "validation_error": validation_error,
                "generation_model": args.model,
                "generated_at_utc": now_utc(),
                "batch_index": call_idx,
            }
            generated.append(record)
            existing_qset.add(q_norm)
            existing_sql_set.add(sql_norm)
            next_numeric_id += 1
            accepted += 1

            if len(generated) >= target_new:
                break

        print(
            f"[INFO] Batch {call_idx}: parsed={len(items)} accepted={accepted} "
            f"new_total={len(generated)}/{target_new} "
            f"global_total={initial_count + len(generated)}/{args.num_questions}"
        )
        print(
            "[INFO] Batch reject stats: "
            f"dup_q={batch_duplicate_question}, "
            f"dup_sql={batch_duplicate_sql}, "
            f"too_technical={batch_too_technical}, "
            f"not_business_like={batch_not_business_like}, "
            f"invalid_sql={batch_invalid_sql}"
        )

        # Checkpoint after each batch to avoid losing progress on transient network failures.
        checkpoint_rows = existing_rows + generated
        write_jsonl(args.out_jsonl, checkpoint_rows, append=False)
        write_csv(args.out_csv, checkpoint_rows, append=False)
        print(
            f"[INFO] Checkpoint saved: {len(checkpoint_rows)} rows "
            f"-> {args.out_jsonl.name}"
        )

    conn.close()

    if not generated:
        raise RuntimeError("No candidates generated.")

    final_rows = existing_rows + generated
    write_jsonl(args.out_jsonl, final_rows, append=False)
    write_csv(args.out_csv, final_rows, append=False)
    print(f"[OK] Wrote JSONL: {args.out_jsonl} ({len(final_rows)} rows)")
    print(f"[OK] Wrote CSV:   {args.out_csv} ({len(final_rows)} rows)")
    print(
        "[INFO] Filter stats: "
        f"duplicate_question={filtered_duplicate_question}, "
        f"duplicate_sql={filtered_duplicate_sql}, "
        f"too_technical={filtered_too_technical}, "
        f"not_business_like={filtered_not_business_like}, "
        f"invalid_sql={filtered_invalid_sql}"
    )

    if len(generated) < target_new:
        print(
            "[WARN] Target not fully reached. "
            f"generated_new={len(generated)} requested_new={target_new}"
        )


if __name__ == "__main__":
    main()
