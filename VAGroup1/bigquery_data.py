import streamlit as st
from bigquery_client import get_bigquery_client

@st.cache_data(ttl=3600)
def load_data():
    client = get_bigquery_client()

    query = """
        SELECT *
        FROM `grocery-stockpiling-explorer.grocery_data.enriched_transactions`
    """

    df = client.query(query).to_dataframe()

    return df
