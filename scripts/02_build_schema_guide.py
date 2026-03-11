import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "raw" / "db.sqlite"
OUT_PATH = ROOT / "assets" / "schema_guide.yaml"
BUSINESS_CONTEXT_PATH = ROOT / "assets" / "schema_business_context.json"


def load_business_context(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Business context JSON must be an object.")
    return data


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value)
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def to_yaml(obj: Any, indent: int = 0) -> str:
    pad = " " * indent

    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(to_yaml(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(value)}")
        return "\n".join(lines)

    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return "\n".join(lines)

    return f"{pad}{yaml_scalar(obj)}"


def table_names(conn: sqlite3.Connection, include_staging: bool) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
    ).fetchall()
    names = [r[0] for r in rows]
    if include_staging:
        return names
    return [n for n in names if not n.startswith("stg_")]


def fk_map(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, str]]:
    rows = conn.execute(f'PRAGMA foreign_key_list("{table}");').fetchall()
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        from_col = r[3]
        out[from_col] = {"to_table": r[2], "to_column": r[4]}
    return out


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


def sample_values(
    conn: sqlite3.Connection, table: str, column: str, sample_size: int
) -> list[Any]:
    query = (
        f'SELECT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL '
        f'GROUP BY "{column}" '
        f'ORDER BY COUNT(*) DESC, "{column}" '
        f"LIMIT {sample_size};"
    )
    rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]


def table_profile(
    conn: sqlite3.Connection,
    table: str,
    sample_size: int,
    business_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_rows = conn.execute(f'SELECT COUNT(*) FROM "{table}";').fetchone()[0]
    pragma_cols = conn.execute(f'PRAGMA table_info("{table}");').fetchall()
    fks = fk_map(conn, table)

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

    columns = []
    primary_key = []
    column_business = {}
    if business_table:
        raw_col_desc = business_table.get("columns", {})
        if isinstance(raw_col_desc, dict):
            column_business = raw_col_desc

    for col in pragma_cols:
        col_name = col[1]
        col_type = col[2] or "TEXT"
        not_null = bool(col[3])
        default = col[4]
        is_pk = bool(col[5])
        if is_pk:
            primary_key.append(col_name)

        null_count = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL;'
        ).fetchone()[0]
        distinct_count = conn.execute(
            f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}";'
        ).fetchone()[0]

        col_info: dict[str, Any] = {
            "name": col_name,
            "type": col_type,
            "not_null": not_null,
            "default": default,
            "is_primary_key": is_pk,
            "is_foreign_key": col_name in fks,
            "null_count": null_count,
            "distinct_count": distinct_count,
            "sample_values": sample_values(conn, table, col_name, sample_size),
        }

        if col_name in fks:
            col_info["references"] = fks[col_name]
        if col_name in check_values:
            col_info["allowed_values_from_check"] = check_values[col_name]
        if col_name in column_business:
            col_info["business_description"] = column_business[col_name]

        columns.append(col_info)

    profile = {
        "name": table,
        "row_count": total_rows,
        "primary_key": primary_key,
        "columns": columns,
    }

    if business_table:
        if "role" in business_table:
            profile["business_role"] = business_table["role"]
        notes = business_table.get("notes", [])
        if isinstance(notes, list) and notes:
            profile["business_notes"] = notes
        rels = business_table.get("relationships", [])
        if isinstance(rels, list) and rels:
            profile["business_relationships"] = rels
        query_hints = business_table.get("query_hints", [])
        if isinstance(query_hints, list) and query_hints:
            profile["query_hints"] = query_hints
        if "empty_table_is_normal" in business_table:
            profile["empty_table_is_normal"] = business_table["empty_table_is_normal"]

    return profile


def build_schema_guide(
    db_path: Path,
    out_path: Path,
    sample_size: int,
    include_staging: bool,
    business_context: dict[str, Any] | None = None,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    names = table_names(conn, include_staging=include_staging)
    business_tables = {}
    if business_context and isinstance(business_context.get("tables"), dict):
        business_tables = business_context["tables"]

    tables = [
        table_profile(conn, t, sample_size, business_table=business_tables.get(t))
        for t in names
    ]

    guide = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_path": str(db_path),
        "tables_count": len(tables),
        "tables": tables,
    }
    if business_context:
        global_notes = business_context.get("global_notes", [])
        if isinstance(global_notes, list) and global_notes:
            guide["global_business_notes"] = global_notes

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_yaml(guide) + "\n", encoding="utf-8")
    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YAML schema guide from a SQLite database."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite database.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=OUT_PATH,
        help="Output path for schema guide YAML.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Top frequent non-null values to include per column.",
    )
    parser.add_argument(
        "--include-staging",
        action="store_true",
        help="Include stg_* tables in the guide.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print a compact JSON summary to stdout.",
    )
    parser.add_argument(
        "--business-context",
        type=Path,
        default=BUSINESS_CONTEXT_PATH,
        help="Optional JSON file containing business descriptions per table/column.",
    )
    parser.add_argument(
        "--no-business-context",
        action="store_true",
        help="Do not merge business context into the output guide.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    business_context = {}
    if not args.no_business_context:
        business_context = load_business_context(args.business_context)

    build_schema_guide(
        db_path=args.db_path,
        out_path=args.out_path,
        sample_size=max(1, args.sample_size),
        include_staging=args.include_staging,
        business_context=business_context,
    )

    if args.print_json:
        payload = {
            "db_path": str(args.db_path),
            "out_path": str(args.out_path),
            "sample_size": max(1, args.sample_size),
            "include_staging": args.include_staging,
            "business_context_path": (
                None if args.no_business_context else str(args.business_context)
            ),
        }
        print(json.dumps(payload, ensure_ascii=False))

    print(f"[OK] Schema guide generated: {args.out_path}")


if __name__ == "__main__":
    main()
