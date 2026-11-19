import streamlit as st
import pandas as pd
import gc

from bigquery_client import get_bigquery_client


@st.cache_data(ttl=3600)
def load_data():
    client = get_bigquery_client()

    query = """
        SELECT *
        FROM `grocery-stockpiling-explorer.grocery_data.enriched_transactions`
    """

    df = client.query(query).to_dataframe()

    # Reduce memory usage, matching the old Parquet loader
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")

    gc.collect()
    return df
