import os
import sys

import pandas as pd

sys.path.append(os.path.abspath("."))

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