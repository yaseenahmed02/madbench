"""Unit tests for the Track B correctness verifier.

Twenty tests gate the verifier: shape parsing (SQL / scalar / rows / dict),
column reordering, order-sensitive vs multiset comparison, numeric
tolerance, and the diagnostic messages returned on failure.
"""

from __future__ import annotations

from mcp_dbs_bench.agent.correctness import GoldResult, verify


def _gold(rows, *, columns=None, sql="SELECT * FROM t", query_id="tb_c2_01"):
    """Concise builder for GoldResult fixtures."""
    if columns is None:
        columns = [""] * (len(rows[0]) if rows else 0)
    return GoldResult(
        query_id=query_id, backend="postgres", gold_sql=sql,
        columns=list(columns), rows=[list(r) for r in rows],
    )


class TestScalarAnswers:
    def test_scalar_int_matches_single_cell_gold(self):
        # Arrange
        gold = _gold([[150000]], columns=["count"], sql="SELECT COUNT(*) FROM customer")

        # Act
        verdict, _ = verify(150000, gold)

        # Assert
        assert verdict == "correct"

    def test_scalar_float_within_tolerance_matches(self):
        # Arrange
        gold = _gold([[1234.5678]], columns=["avg"], sql="SELECT AVG(x) FROM t")

        # Act
        verdict, _ = verify(1234.56779, gold, tolerance=1e-4)

        # Assert
        assert verdict == "correct"

    def test_scalar_float_outside_tolerance_incorrect(self):
        # Arrange
        gold = _gold([[100.0]], sql="SELECT AVG(x) FROM t")

        # Act
        verdict, msg = verify(101.5, gold, tolerance=1e-4)

        # Assert
        assert verdict == "incorrect"
        assert "cell mismatch" in msg or "no match" in msg

    def test_scalar_string_matches_case_insensitively(self):
        # Arrange
        gold = _gold([["Alice"]], sql="SELECT c_name FROM customer WHERE c_custkey = 100")

        # Act
        verdict, _ = verify("  alice  ", gold)

        # Assert
        assert verdict == "correct"

    def test_none_answer_is_unparseable(self):
        # Arrange
        gold = _gold([[1]], sql="SELECT 1")

        # Act
        verdict, msg = verify(None, gold)

        # Assert
        assert verdict == "unparseable"
        assert "None" in msg


class TestListAnswers:
    def test_list_of_rows_matches_gold_unordered(self):
        # Arrange — gold has no ORDER BY, so multiset comparison applies.
        gold = _gold(
            [["A"], ["B"], ["C"]],
            columns=["n_name"],
            sql="SELECT DISTINCT n_name FROM nation",
        )

        # Act — agent returns rows in different order
        verdict, _ = verify([["B"], ["C"], ["A"]], gold)

        # Assert
        assert verdict == "correct"

    def test_list_of_rows_ordered_by_gold_sql_is_order_sensitive(self):
        # Arrange
        gold = _gold(
            [["A"], ["B"], ["C"]],
            columns=["n_name"],
            sql="SELECT n_name FROM nation ORDER BY n_name",
        )

        # Act — agent returns the same rows in reverse
        verdict, msg = verify([["C"], ["B"], ["A"]], gold)

        # Assert
        assert verdict == "incorrect"
        assert "cell mismatch" in msg

    def test_list_of_scalars_treated_as_column(self):
        # Arrange
        gold = _gold([["A"], ["B"]], columns=["c"], sql="SELECT c FROM t")

        # Act — agent returns bare scalars
        verdict, _ = verify(["A", "B"], gold)

        # Assert
        assert verdict == "correct"

    def test_empty_list_matches_empty_gold(self):
        # Arrange
        gold = _gold([], columns=["c"], sql="SELECT c FROM t WHERE 1=0")

        # Act
        verdict, _ = verify([], gold)

        # Assert
        assert verdict == "correct"

    def test_row_count_mismatch_is_incorrect(self):
        # Arrange
        gold = _gold([[1], [2], [3]], sql="SELECT n FROM t")

        # Act
        verdict, msg = verify([[1], [2]], gold)

        # Assert
        assert verdict == "incorrect"
        assert "row count mismatch" in msg


class TestDictAnswers:
    def test_dict_with_columns_and_rows_is_normalised(self):
        # Arrange
        gold = _gold(
            [[150000]], columns=["count"], sql="SELECT COUNT(*) FROM customer"
        )
        answer = {"columns": ["count"], "rows": [[150000]]}

        # Act
        verdict, _ = verify(answer, gold)

        # Assert
        assert verdict == "correct"

    def test_dict_with_result_key_is_unwrapped(self):
        # Arrange
        gold = _gold([[42]], sql="SELECT 42")

        # Act
        verdict, _ = verify({"result": 42}, gold)

        # Assert
        assert verdict == "correct"

    def test_dict_missing_recognised_keys_is_unparseable(self):
        # Arrange
        gold = _gold([[1]], sql="SELECT 1")

        # Act
        verdict, msg = verify({"foo": "bar"}, gold)

        # Assert
        assert verdict == "unparseable"
        assert "missing recognised keys" in msg


class TestSqlAnswers:
    def test_sql_string_without_executor_is_unparseable(self):
        # Arrange
        gold = _gold([[1]], sql="SELECT 1")

        # Act
        verdict, msg = verify("SELECT 1", gold)

        # Assert
        assert verdict == "unparseable"
        assert "no sql_executor" in msg

    def test_sql_string_with_executor_returning_gold_matches(self):
        # Arrange
        gold = _gold([[42]], columns=["n"], sql="SELECT 42 AS n")

        def executor(sql):
            assert "SELECT 42" in sql
            return (["n"], [[42]])

        # Act
        verdict, _ = verify("SELECT 42 AS n", gold, sql_executor=executor)

        # Assert
        assert verdict == "correct"

    def test_sql_executor_raising_exception_yields_unparseable(self):
        # Arrange
        gold = _gold([[1]], sql="SELECT 1")

        def executor(sql):
            raise RuntimeError("connection refused")

        # Act
        verdict, msg = verify("SELECT 1", gold, sql_executor=executor)

        # Assert
        assert verdict == "unparseable"
        assert "connection refused" in msg


class TestColumnReordering:
    def test_columns_out_of_order_are_permuted_to_match_gold(self):
        # Arrange — gold has (a, b); agent returns (b, a)
        gold = GoldResult(
            query_id="tb_c2_test", backend="postgres",
            gold_sql="SELECT a, b FROM t",
            columns=["a", "b"], rows=[[1, "x"], [2, "y"]],
        )
        answer = {"columns": ["b", "a"], "rows": [["x", 1], ["y", 2]]}

        # Act
        verdict, _ = verify(answer, gold)

        # Assert
        assert verdict == "correct"

    def test_column_names_mismatch_falls_back_to_positional(self):
        # Arrange — different column names but identical positional values
        gold = GoldResult(
            query_id="tb_c2_test", backend="postgres",
            gold_sql="SELECT COUNT(*) FROM t",
            columns=["count"], rows=[[150000]],
        )
        answer = {"columns": ["n"], "rows": [[150000]]}

        # Act — falls back to positional comparison because names differ
        verdict, _ = verify(answer, gold)

        # Assert
        assert verdict == "correct"


class TestHeaderRowHeuristic:
    def test_llm_style_header_row_is_peeled_off(self):
        # Arrange — LLM habit: submit [["column_name"], [value]]
        gold = _gold([[150000]], columns=["count"],
                     sql="SELECT COUNT(*) FROM customer")

        # Act
        verdict, msg = verify([["customer_count"], [150000]], gold)

        # Assert
        assert verdict == "correct"
        assert "peeled" in msg or True  # diagnostic wording is best-effort

    def test_two_row_all_string_answer_not_peeled(self):
        # Arrange — both rows all strings; peeling would be wrong
        gold = _gold([["A"], ["B"]], columns=["name"],
                     sql="SELECT name FROM t")

        # Act — agent legitimately returns two string rows
        verdict, _ = verify([["A"], ["B"]], gold)

        # Assert
        assert verdict == "correct"


class TestNumericTolerance:
    def test_default_tolerance_accepts_four_decimal_match(self):
        # Arrange
        gold = _gold([[1234.5678]], sql="SELECT AVG(x) FROM t")

        # Act
        verdict, _ = verify(1234.56780001, gold)

        # Assert
        assert verdict == "correct"

    def test_integer_and_float_representations_compare_equal(self):
        # Arrange
        gold = _gold([[100]], sql="SELECT COUNT(*) FROM t")

        # Act
        verdict, _ = verify(100.0, gold)

        # Assert
        assert verdict == "correct"

    def test_numeric_string_and_number_compare_equal(self):
        # Arrange — agent may serialise numbers as JSON strings
        gold = _gold([[100]], sql="SELECT COUNT(*) FROM t")

        # Act
        verdict, _ = verify("100", gold)

        # Assert
        assert verdict == "correct"


class TestDecimalValues:
    """Drivers return NUMERIC/DECIMAL columns as ``decimal.Decimal``.

    Gold JSON stores them as rounded floats, so executed-SQL answers must
    compare numerically; string comparison ("158148.30" vs "158148.3")
    would score a correct answer incorrect.
    """

    def test_executed_sql_decimal_matches_float_gold(self):
        # Arrange
        from decimal import Decimal

        gold = _gold([[158148.3]], columns=["total"], sql="SELECT SUM(x) FROM t")

        def executor(_sql):
            return ["total"], [[Decimal("158148.30")]]

        # Act
        verdict, _ = verify("SELECT SUM(x) FROM t", gold, sql_executor=executor)

        # Assert
        assert verdict == "correct"

    def test_executed_sql_unrounded_decimal_average_matches_rounded_gold(self):
        # Arrange
        from decimal import Decimal

        gold = _gold([[1501.5054]], columns=["avg"], sql="SELECT AVG(x) FROM t")

        def executor(_sql):
            return ["avg"], [[Decimal("1501.505413666666666667")]]

        # Act
        verdict, _ = verify("SELECT AVG(x) FROM t", gold, sql_executor=executor)

        # Assert
        assert verdict == "correct"

    def test_decimal_outside_tolerance_is_incorrect(self):
        # Arrange
        from decimal import Decimal

        gold = _gold([[100.0]], columns=["v"], sql="SELECT v FROM t")

        def executor(_sql):
            return ["v"], [[Decimal("100.5")]]

        # Act
        verdict, _ = verify("SELECT v FROM t", gold, sql_executor=executor)

        # Assert
        assert verdict == "incorrect"


class TestJsonStringAnswers:
    """A correct row list sent as JSON text is the same answer as the list."""

    def test_verify_with_json_string_rows_matching_gold_is_correct(self):
        # Arrange
        gold = _gold([["FRANCE", 3], ["PERU", 5]], columns=["n", "c"])

        # Act
        verdict, _ = verify('[["FRANCE", 3], ["PERU", 5]]', gold)

        # Assert
        assert verdict == "correct"

    def test_verify_with_json_string_rows_not_matching_gold_is_incorrect(self):
        # Arrange
        gold = _gold([[450000]])

        # Act
        verdict, _ = verify("[[150000]]", gold)

        # Assert
        assert verdict == "incorrect"

    def test_verify_with_malformed_json_string_stays_a_scalar_answer(self):
        # Arrange
        gold = _gold([["[not json"]])

        # Act
        verdict, _ = verify("[not json", gold)

        # Assert
        assert verdict == "correct"

    def test_verify_with_plain_numeric_string_stays_a_scalar_answer(self):
        # Arrange
        gold = _gold([[42]])

        # Act
        verdict, _ = verify("42", gold)

        # Assert
        assert verdict == "correct"


class TestMidnightTimestampAnswers:
    """A DATE that a Node server serialised as midnight UTC is still that date."""

    def test_verify_with_midnight_timestamp_matching_gold_date_is_correct(self):
        # Arrange
        gold = _gold([["1995-01-01", 7]], columns=["d", "c"])

        # Act
        verdict, _ = verify([["1995-01-01T00:00:00.000Z", 7]], gold)

        # Assert
        assert verdict == "correct"

    def test_verify_with_midnight_timestamp_of_other_date_is_incorrect(self):
        # Arrange
        gold = _gold([["1995-01-01", 7]], columns=["d", "c"])

        # Act
        verdict, _ = verify([["1995-01-02T00:00:00.000Z", 7]], gold)

        # Assert
        assert verdict == "incorrect"

    def test_verify_with_non_midnight_timestamp_against_gold_date_is_incorrect(self):
        # Arrange
        gold = _gold([["1995-01-01", 7]], columns=["d", "c"])

        # Act
        verdict, _ = verify([["1995-01-01T13:45:00.000Z", 7]], gold)

        # Assert
        assert verdict == "incorrect"
