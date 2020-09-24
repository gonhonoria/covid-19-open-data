# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import json
from pathlib import Path
from sqlite3.dbapi2 import Connection, Cursor, connect
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pandas import Int64Dtype

from .memory_efficient import read_lines

_SCHEMA_TABLE_NAME = "_table_schemas"
_SCHEMA_TABLE_SCHEMA = {"table_name": "TEXT", "schema_json": "TEXT"}


def _dtype_to_sql_type(dtype: Any) -> str:
    """
    Parse a dtype name and output the equivalent SQL type.

    Arguments:
        dtype: dtype object.
    Returns:
        str: SQL type name.
    """

    if dtype == "str" or dtype == str:
        return "TEXT"
    if dtype == "float" or dtype == float:
        return "DOUBLE"
    if dtype == "int" or dtype == int or isinstance(dtype, Int64Dtype):
        return "INTEGER"
    raise TypeError(f"Unsupported dtype: {dtype}")


def _safe_column_name(column_name: str) -> str:
    if "." in column_name:
        column_name = f"[{column_name}]"
    if column_name in ("on", "left", "right", "index"):
        column_name = f"_{column_name}"
    return column_name


def _safe_table_name(table_name: str) -> str:
    if "." in table_name:
        table_name = f"[{table_name}]"
    if "-" in table_name:
        table_name = table_name.replace("-", "_")
    if table_name in ("on", "left", "right", "index"):
        table_name = f"_{table_name}"
    return table_name


def _statement_insert_record_tuple(
    conn: Connection, table_name: str, columns: Tuple[str], record: Tuple[str]
) -> None:
    placeholders = ", ".join("?" for _ in columns)
    column_names = ", ".join(_safe_column_name(name) for name in columns)
    conn.execute(
        f"INSERT INTO {_safe_table_name(table_name)} ({column_names}) VALUES ({placeholders})",
        record,
    )


def _statement_insert_record_dict(
    conn: Connection, table_name: str, record: Dict[str, str]
) -> None:
    header = tuple(record.keys())
    placeholders = ", ".join("?" for _ in header)
    column_names = ", ".join(_safe_column_name(name) for name in header)
    conn.execute(
        f"INSERT INTO {_safe_table_name(table_name)} ({column_names}) VALUES ({placeholders})",
        tuple(record.values()),
    )


def _fetch_table_schema(conn: Connection, table_name: str) -> Dict[str, str]:
    with conn:
        schema_json = conn.execute(
            f'SELECT schema_json FROM {_SCHEMA_TABLE_NAME} WHERE table_name = "{table_name}"'
        ).fetchone()[0]
        return json.loads(schema_json)


def _output_named_records(cursor: Cursor) -> Iterable[Dict[str, Any]]:
    names = [description[0] for description in cursor.description]
    for record in cursor:
        yield {name: value for name, value in zip(names, record)}
    cursor.close()


def table_create(conn: Connection, table_name: str, schema: Dict[str, str]) -> None:
    sql_schema = ", ".join(f"{_safe_column_name(name)} {dtype}" for name, dtype in schema.items())
    with conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {_safe_table_name(table_name)} ({sql_schema})")

        # Every time a table is created, store its schema in our helper table
        _statement_insert_record_dict(
            conn, _SCHEMA_TABLE_NAME, {"table_name": table_name, "schema_json": json.dumps(schema)}
        )


def create_sqlite_database(db_file: str = ":memory:") -> Connection:
    """
    Creates an SQLite database at the specified location, importing all files from the tables
    folder.
    """
    with connect(db_file) as conn:
        # Create a helper table used to store schemas so we can retrieve them later
        table_create(conn, _SCHEMA_TABLE_NAME, _SCHEMA_TABLE_SCHEMA)
        return conn


def table_import_from_file(
    conn: Connection, table_path: Path, table_name: str = None, schema: Dict[str, str] = None
) -> None:
    """
    Import table from CSV file located at `table_path` using the provided schema for types.

    Arguments:
        cursor: Cursor for the database execution engine
        table_path: Path to the input CSV file
        schema: Pipeline schema for this table
    """
    with conn:

        # Derive table name from file name and open a CSV reader
        reader = csv.reader(read_lines(table_path))
        table_name = _safe_table_name(table_name or table_path.stem)

        #
        header = list(next(reader))
        schema = schema or {}

        sql_schema = {}
        for name in header:
            sql_schema[name] = _dtype_to_sql_type(schema.get(name, "str"))

        table_create(conn, table_name, sql_schema)

        header = sql_schema.keys()
        for record in reader:
            _statement_insert_record_tuple(conn, table_name, header, record)


def table_import_from_records(
    conn: Connection,
    table_name: str,
    records: Iterable[Dict[str, Any]],
    schema: Dict[str, str] = None,
) -> None:
    schema = schema or {}

    with conn:
        # Read the first record to derive the header
        if isinstance(records, list):
            first_record = records[0]
            records = records[1:]
        else:
            first_record = next(records)

        # Get the SQL schema from a combination of the record and the provided schema
        sql_schema = {}
        for name in first_record.keys():
            sql_schema[name] = _dtype_to_sql_type(schema.get(name, "str"))

        # Create the table in the db and insert the first record
        table_create(conn, table_name, sql_schema)
        _statement_insert_record_dict(conn, table_name, first_record)

        # Insert all remaining records
        for record in records:
            _statement_insert_record_dict(conn, table_name, record)


def table_select_all(conn: Connection, table_name: str) -> Iterable[Dict[str, Any]]:
    table_name = _safe_table_name(table_name)
    cursor = conn.execute(f"SELECT * FROM {table_name}")
    yield from _output_named_records(cursor)
    cursor.close()


def table_merge(
    conn: Connection,
    left: str,
    right: str,
    on: List[str],
    how: str = "inner",
    into_table: str = None,
) -> Optional[Iterable[Dict[str, Any]]]:

    left = _safe_table_name(left)
    right = _safe_table_name(right)
    assert left != right, f"Tables must have unique names, found {left} for both"

    clause_on = " AND ".join(f"{left}.{col} = {right}.{col}" for col in on)
    statement_join = f"SELECT * FROM {left} {how} JOIN {right} ON ({clause_on})"

    # If we are inserting the result into another table, create it first and prepare statement
    if into_table:
        combined_schema = {}
        combined_schema.update(_fetch_table_schema(conn, left))
        combined_schema.update(_fetch_table_schema(conn, right))

        into_table = _safe_table_name(into_table)
        with conn:
            table_create(conn, into_table, combined_schema)

    # Otherwise perform the merge and yield records from the cursor
    else:
        with conn:
            cursor = conn.execute(statement_join)
            return _output_named_records(cursor)


def table_multimerge(
    conn: Connection,
    table_names: List[str],
    on: List[str],
    how: str = "inner",
    into_table: str = None,
) -> Optional[Iterable[Dict[str, Any]]]:

    table_names = [_safe_table_name(name) for name in table_names]
    assert len(table_names) == len(set(table_names)), f"Table names must all be unique"

    left = table_names[0]
    statement_join = f"SELECT * FROM {left}"
    for right in table_names[1:]:
        clause_on = " AND ".join(f"{left}.{col} = {right}.{col}" for col in on)
        statement_join += f" {how} JOIN {right} ON ({clause_on})"

    if into_table:
        combined_schema = {}
        for table_name in table_names:
            combined_schema.update(_fetch_table_schema(conn, table_name))

        into_table = _safe_table_name(into_table)
        with conn:
            table_create(conn, into_table, combined_schema)

    # Otherwise perform the merge and yield records from the cursor
    else:
        cursor = conn.execute(statement_join)
        return _output_named_records(cursor)


def table_as_csv(conn: Connection, table_name: str, output_path: Path = None) -> None:
    with open(output_path, "w") as fd:
        writer = csv.writer(fd)
        records = table_select_all(conn, table_name)
        first_record = next(records)
        writer.writerow(first_record.keys())
        writer.writerow(first_record.values())
        writer.writerows(records)
