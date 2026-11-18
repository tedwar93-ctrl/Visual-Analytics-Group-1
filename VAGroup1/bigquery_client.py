
import json
import streamlit as st
from google.cloud import bigquery


@st.cache_resource
def get_bigquery_client():
    credentials_info = json.loads(
        st.secrets["gcp_service_account"]
    )

    return bigquery.Client.from_service_account_info(
        credentials_info
    )
