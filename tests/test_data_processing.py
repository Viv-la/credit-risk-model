import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd

from src.data_processing import calculate_rfm


def test_calculate_rfm_returns_dataframe():
    data = {
        "CustomerId": ["A", "A", "B"],
        "TransactionId": ["T1", "T2", "T3"],
        "TransactionStartTime": [
            "2018-11-01",
            "2018-11-02",
            "2018-11-03",
        ],
        "Value": [100, 200, 300],
    }

    df = pd.DataFrame(data)

    result = calculate_rfm(df)

    assert result is not None


def test_calculate_rfm_has_expected_columns():
    data = {
        "CustomerId": ["A", "A", "B"],
        "TransactionId": ["T1", "T2", "T3"],
        "TransactionStartTime": [
            "2018-11-01",
            "2018-11-02",
            "2018-11-03",
        ],
        "Value": [100, 200, 300],
    }

    df = pd.DataFrame(data)

    result = calculate_rfm(df)

    assert "recency" in result.columns
    assert "frequency" in result.columns
    assert "monetary" in result.columns