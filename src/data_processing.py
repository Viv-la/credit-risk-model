import pandas as pd


def load_raw_data(path):
    """Load raw transaction data."""
    return pd.read_csv(path)


def create_aggregate_features(df):
    """Create customer-level aggregate transaction features."""
    aggregate_df = df.groupby("CustomerId").agg(
        total_transaction_amount=("Amount", "sum"),
        average_transaction_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_transaction_amount=("Amount", "std"),
        total_transaction_value=("Value", "sum"),
        average_transaction_value=("Value", "mean"),
    ).reset_index()

    aggregate_df["std_transaction_amount"] = aggregate_df[
        "std_transaction_amount"
    ].fillna(0)

    return aggregate_df


def create_time_features(df):
    """Extract transaction time features."""
    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df

def calculate_rfm(df):
    """Calculate Recency, Frequency, and Monetary metrics per customer."""
    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Value", "sum"),
    ).reset_index()

    return rfm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def create_high_risk_label(rfm):
    """Cluster customers using RFM and assign high-risk proxy label."""
    rfm = rfm.copy()

    features = rfm[["recency", "frequency", "monetary"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(scaled_features)

    cluster_summary = rfm.groupby("cluster").agg(
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        customer_count=("CustomerId", "count")
    ).reset_index()

    high_risk_cluster = cluster_summary.sort_values(
        by=["avg_frequency", "avg_monetary"],
        ascending=[True, True]
    ).iloc[0]["cluster"]

    rfm["is_high_risk"] = (
        rfm["cluster"] == high_risk_cluster
    ).astype(int)

    return rfm, cluster_summary
from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(df):
    """Encode categorical variables."""
    
    df = df.copy()

    categorical_columns = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    encoders = {}

    for col in categorical_columns:
        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder

    return df
def save_processed_data(df, path):
    """Save processed dataset."""
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")