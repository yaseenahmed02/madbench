"""TPC-H schema DDL per backend (TPC-H specification rev 3.0.1, section 1.4).

Each dialect's CREATE TABLE statements appear here so the harness can load
SF1/SF3/SF10 data into PG / MySQL / DuckDB without external schema files.

The schema is public (TPC published it in 1999); only the *queries* require
attribution. The integer types match the TPC-H spec's IDENTIFIER (8-byte
signed) and DECIMAL(15,2) for prices.
"""

from __future__ import annotations

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS region (
        r_regionkey  INTEGER PRIMARY KEY,
        r_name       CHAR(25) NOT NULL,
        r_comment    VARCHAR(152)
    )""",
    """CREATE TABLE IF NOT EXISTS nation (
        n_nationkey  INTEGER PRIMARY KEY,
        n_name       CHAR(25) NOT NULL,
        n_regionkey  INTEGER NOT NULL REFERENCES region(r_regionkey),
        n_comment    VARCHAR(152)
    )""",
    """CREATE TABLE IF NOT EXISTS part (
        p_partkey     INTEGER PRIMARY KEY,
        p_name        VARCHAR(55) NOT NULL,
        p_mfgr        CHAR(25) NOT NULL,
        p_brand       CHAR(10) NOT NULL,
        p_type        VARCHAR(25) NOT NULL,
        p_size        INTEGER NOT NULL,
        p_container   CHAR(10) NOT NULL,
        p_retailprice DECIMAL(15,2) NOT NULL,
        p_comment     VARCHAR(23) NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS supplier (
        s_suppkey     INTEGER PRIMARY KEY,
        s_name        CHAR(25) NOT NULL,
        s_address     VARCHAR(40) NOT NULL,
        s_nationkey   INTEGER NOT NULL REFERENCES nation(n_nationkey),
        s_phone       CHAR(15) NOT NULL,
        s_acctbal     DECIMAL(15,2) NOT NULL,
        s_comment     VARCHAR(101) NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS partsupp (
        ps_partkey     INTEGER NOT NULL REFERENCES part(p_partkey),
        ps_suppkey     INTEGER NOT NULL REFERENCES supplier(s_suppkey),
        ps_availqty    INTEGER NOT NULL,
        ps_supplycost  DECIMAL(15,2) NOT NULL,
        ps_comment     VARCHAR(199) NOT NULL,
        PRIMARY KEY (ps_partkey, ps_suppkey)
    )""",
    """CREATE TABLE IF NOT EXISTS customer (
        c_custkey     INTEGER PRIMARY KEY,
        c_name        VARCHAR(25) NOT NULL,
        c_address     VARCHAR(40) NOT NULL,
        c_nationkey   INTEGER NOT NULL REFERENCES nation(n_nationkey),
        c_phone       CHAR(15) NOT NULL,
        c_acctbal     DECIMAL(15,2) NOT NULL,
        c_mktsegment  CHAR(10) NOT NULL,
        c_comment     VARCHAR(117) NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS orders (
        o_orderkey      INTEGER PRIMARY KEY,
        o_custkey       INTEGER NOT NULL REFERENCES customer(c_custkey),
        o_orderstatus   CHAR(1) NOT NULL,
        o_totalprice    DECIMAL(15,2) NOT NULL,
        o_orderdate     DATE NOT NULL,
        o_orderpriority CHAR(15) NOT NULL,
        o_clerk         CHAR(15) NOT NULL,
        o_shippriority  INTEGER NOT NULL,
        o_comment       VARCHAR(79) NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS lineitem (
        l_orderkey       INTEGER NOT NULL REFERENCES orders(o_orderkey),
        l_partkey        INTEGER NOT NULL REFERENCES part(p_partkey),
        l_suppkey        INTEGER NOT NULL REFERENCES supplier(s_suppkey),
        l_linenumber     INTEGER NOT NULL,
        l_quantity       DECIMAL(15,2) NOT NULL,
        l_extendedprice  DECIMAL(15,2) NOT NULL,
        l_discount       DECIMAL(15,2) NOT NULL,
        l_tax            DECIMAL(15,2) NOT NULL,
        l_returnflag     CHAR(1) NOT NULL,
        l_linestatus     CHAR(1) NOT NULL,
        l_shipdate       DATE NOT NULL,
        l_commitdate     DATE NOT NULL,
        l_receiptdate    DATE NOT NULL,
        l_shipinstruct   CHAR(25) NOT NULL,
        l_shipmode       CHAR(10) NOT NULL,
        l_comment        VARCHAR(44) NOT NULL,
        PRIMARY KEY (l_orderkey, l_linenumber)
    )""",
]

# MySQL: same shape but uses VARCHAR for CHAR fields to avoid padding
# differences, and InnoDB ENGINE so FKs are enforced.
_MYSQL_DDL = [
    s.replace("PRIMARY KEY", "PRIMARY KEY").replace("CHAR(", "CHAR(") + " ENGINE=InnoDB"
    for s in _PG_DDL
]

# DuckDB: identical to PG except FK references can be omitted; embedded engine
# already enforces type constraints and avoids the FK ordering hassle of bulk loads.
_DUCKDB_DDL = [
    s.replace("REFERENCES region(r_regionkey)", "")
    .replace("REFERENCES nation(n_nationkey)", "")
    .replace("REFERENCES part(p_partkey)", "")
    .replace("REFERENCES supplier(s_suppkey)", "")
    .replace("REFERENCES customer(c_custkey)", "")
    .replace("REFERENCES orders(o_orderkey)", "")
    for s in _PG_DDL
]


SCHEMA_DDL: dict[str, list[str]] = {
    "postgres": _PG_DDL,
    "mysql": _MYSQL_DDL,
    "duckdb": _DUCKDB_DDL,
}

TPCH_TABLES: tuple[str, ...] = (
    "region",
    "nation",
    "part",
    "supplier",
    "partsupp",
    "customer",
    "orders",
    "lineitem",
)
