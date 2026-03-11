import sqlite3
from pathlib import Path
import pandas as pd
import csv

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RAW = ROOT / "data" / "raw"
INTER = ROOT / "data" / "intermediate"
REJECTS = INTER / "rejects"

CSV_DIR = RAW / "pgadmin_exports"
DB_PATH = RAW / "db.sqlite"
SCHEMA_PATH = ASSETS / "schema_sqlite.sql"

TABLES_IN_ORDER = [
    "states",
    "rulesets",
    "documents",
    "extraction_runs",
    "extraction_results",
    "resultfeedbacks",
    "sections",
    "field_rules",
    "validation_rules",
]

# Some pgAdmin CSV exports do not follow the SQLite table column order.
# Key = table name, value = CSV column order as exported.
CSV_COLUMN_ORDER_OVERRIDES = {
    "documents": [
        "document_id",
        "mask",
        "siret",
        "city_name",
        "format",
        "document_year",
        "created_at",
        "link_bucketS3",
        "state_id",
        "error_message",
        "nb_pages",
        "warning_message",
        "budget_type",
    ],
}

def load_schema(conn: sqlite3.Connection) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)

def expected_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [r[1] for r in rows]

def file_has_any_data_line(csv_path: Path) -> bool:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            if line.strip():
                return True
    return False

def _read_csv_noheader_strict(csv_path: Path, exp_cols: list[str]) -> list[list[str]]:
    """
    Ultra-robust no-header CSV reader using Python's csv module.
    - No line skipping.
    - Handles quoted fields with commas/newlines better than pandas C engine in messy exports.
    - Tries delimiter ',' then ';'.
    """
    def read_with_delim(delim: str):
        rows = []
        with open(csv_path, "r", encoding="utf-8", errors="strict", newline="") as f:
            reader = csv.reader(
                f,
                delimiter=delim,
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                strict=True,
            )
            for i, row in enumerate(reader, start=1):
                # allow empty trailing lines
                if len(row) == 0 or (len(row) == 1 and row[0].strip() == ""):
                    continue
                if len(row) != len(exp_cols):
                    raise ValueError(
                        f"[{csv_path.name}] Bad row at line {i}: got {len(row)} fields, expected {len(exp_cols)}.\n"
                        f"Row sample (first 5 fields): {row[:5]}"
                    )
                rows.append(row)
        return rows

    errors = []
    for delim in (",", ";"):
        try:
            return read_with_delim(delim)
        except Exception as e:
            errors.append(f"{delim}: {e}")

    raise ValueError(
        f"[{csv_path.name}] strict parser failed for both delimiters. "
        + " | ".join(errors)
    )

def _split_weird_csv_line(line: str, delimiter: str = ",") -> list[str]:
    """
    Split a CSV line while tolerating pgAdmin-style payloads containing `'"`.
    In these payloads, internal `"` are not escaped with standard CSV rules.
    """
    fields = []
    buf = []
    in_quotes = False
    i = 0
    n = len(line)

    while i < n:
        ch = line[i]

        if ch == '"':
            prev = line[i - 1] if i > 0 else ""
            nxt = line[i + 1] if i + 1 < n else ""

            # Standard escaped quote in CSV
            if in_quotes and nxt == '"':
                buf.append('"')
                i += 2
                continue

            # Internal quote encoded as `'"`
            if prev == "'":
                buf.append('"')
                i += 1
                continue

            in_quotes = not in_quotes
            i += 1
            continue

        if ch == delimiter and not in_quotes:
            fields.append("".join(buf))
            buf = []
        else:
            buf.append(ch)

        i += 1

    fields.append("".join(buf))
    return fields

def _read_csv_noheader_tolerant(
    csv_path: Path, exp_cols: list[str]
) -> tuple[list[list[str]], list[dict]]:
    """
    Fallback parser for malformed CSV rows.
    Parses line-by-line so one broken quote does not abort the whole file.
    """
    rows = []
    rejects = []
    expected = len(exp_cols)

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            parsed_row = None
            best_row = None

            for delim in (",", ";"):
                candidate = _split_weird_csv_line(line, delimiter=delim)
                if best_row is None or abs(len(candidate) - expected) < abs(
                    len(best_row) - expected
                ):
                    best_row = candidate

                if len(candidate) == expected:
                    parsed_row = candidate
                    break

            if parsed_row is not None:
                rows.append(parsed_row)
                continue

            # Keep traceability for malformed rows.
            reject = {
                "error": (
                    f"Parse mismatch at line {line_no}: got "
                    f"{len(best_row) if best_row is not None else 'n/a'} fields, "
                    f"expected {expected}."
                ),
                "__line_no": line_no,
                "__raw_line": line,
            }
            if best_row:
                for c, v in zip(exp_cols, best_row):
                    reject[c] = v
            rejects.append(reject)

    return rows, rejects

def read_csv_noheader(csv_path: Path, exp_cols: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    try:
        rows = _read_csv_noheader_strict(csv_path, exp_cols)
        return pd.DataFrame(rows, columns=exp_cols), []
    except Exception as e:
        print(
            f"[WARN] {csv_path.name}: strict parser failed ({e}). "
            "Falling back to tolerant parser."
        )
        rows, parse_rejects = _read_csv_noheader_tolerant(csv_path, exp_cols)
        return pd.DataFrame(rows, columns=exp_cols), parse_rejects

def csv_columns_for_table(table: str, table_cols: list[str]) -> list[str]:
    override = CSV_COLUMN_ORDER_OVERRIDES.get(table)
    if not override:
        return table_cols

    if set(override) != set(table_cols):
        raise ValueError(
            f"CSV column override for table '{table}' does not match table columns."
        )
    return override

def create_staging_table(conn: sqlite3.Connection, table: str, cols: list[str]) -> str:
    stg = f"stg_{table}"
    conn.execute(f'DROP TABLE IF EXISTS "{stg}";')
    cols_sql = ", ".join([f'"{c}" TEXT' for c in cols])
    conn.execute(f'CREATE TABLE "{stg}" ({cols_sql});')
    return stg

def insert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame):
    df = df.where(pd.notnull(df), None)
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=2000)

def try_insert_row(conn: sqlite3.Connection, table: str, cols: list[str], row: list):
    placeholders = ",".join(["?"] * len(cols))
    collist = ",".join([f'"{c}"' for c in cols])
    conn.execute(f'INSERT INTO "{table}" ({collist}) VALUES ({placeholders});', row)

def write_rejects(table: str, cols: list[str], rejects: list[dict]):
    if not rejects:
        return

    REJECTS.mkdir(parents=True, exist_ok=True)
    out = REJECTS / f"{table}.csv"

    base_fields = ["error"] + cols
    extras = []
    seen = set(base_fields)
    for r in rejects:
        for key in r.keys():
            if key not in seen:
                seen.add(key)
                extras.append(key)

    fieldnames = base_fields + extras
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rejects:
            w.writerow(r)
    print(f"[WARN] {table}: wrote rejects -> {out} ({len(rejects)} rows)")

def load_table_via_staging(conn: sqlite3.Connection, table: str):
    csv_path = CSV_DIR / f"{table}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    table_cols = expected_columns(conn, table)
    csv_cols = csv_columns_for_table(table, table_cols)

    if csv_path.stat().st_size == 0 or not file_has_any_data_line(csv_path):
        print(f"[INFO] {table}: CSV empty -> 0 rows")
        return

    df, parse_rejects = read_csv_noheader(csv_path, csv_cols)
    if csv_cols != table_cols:
        # Normalize to SQLite column order before staging insertion.
        df = df.reindex(columns=table_cols)
    print(
        f"[INFO] {table}: loaded shape={df.shape} from {csv_path.name} "
        f"(parse rejects: {len(parse_rejects)})"
    )

    if df.empty:
        print(f"[INFO] {table}: no valid rows to import")
        write_rejects(table, table_cols, parse_rejects)
        return

    # 1) staging (no constraints)
    stg = create_staging_table(conn, table, table_cols)
    insert_df(conn, stg, df)

    # 2) try bulk insert into real table (fast path)
    conn.execute(f'DELETE FROM "{table}";')  # ensure idempotent per run
    try:
        conn.execute(f'INSERT INTO "{table}" SELECT * FROM "{stg}";')
        print(f"[OK] Imported {table}: {len(df)} rows (bulk)")
        write_rejects(table, table_cols, parse_rejects)
        return
    except sqlite3.IntegrityError as e:
        # Need row-by-row to capture rejects
        print(f"[WARN] {table}: bulk insert failed -> row-wise capture. Reason: {e}")

    # 3) row-wise insert with rejects capture (no loss)
    rejects = list(parse_rejects)
    inserted = 0
    cur = conn.execute(f'SELECT * FROM "{stg}";')
    for row in cur:
        row_list = list(row)
        try:
            try_insert_row(conn, table, table_cols, row_list)
            inserted += 1
        except sqlite3.IntegrityError as e:
            rej = {"error": str(e)}
            for c, v in zip(table_cols, row_list):
                rej[c] = v
            rejects.append(rej)

    print(f"[OK] Imported {table}: {inserted} rows (row-wise)")
    write_rejects(table, table_cols, rejects)

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    INTER.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON;")

    load_schema(conn)

    for t in TABLES_IN_ORDER:
        load_table_via_staging(conn, t)

    conn.commit()
    conn.close()

    print(f"\n[OK] SQLite DB ready: {DB_PATH}")
    if REJECTS.exists():
        print(f"[INFO] Rejects folder: {REJECTS}")

if __name__ == "__main__":
    main()
