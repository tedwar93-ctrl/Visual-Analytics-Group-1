import os
import pandas as pd

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")

print(f"Loading datasets from: {DATA_DIR}...")
transactions = pd.read_csv(os.path.join(DATA_DIR, "transaction_data.csv"))
products = pd.read_csv(os.path.join(DATA_DIR, "product.csv"))
demographics = pd.read_csv(os.path.join(DATA_DIR, "hh_demographic.csv"))
campaign_desc = pd.read_csv(os.path.join(DATA_DIR, "campaign_desc.csv"))
campaign_table = pd.read_csv(os.path.join(DATA_DIR, "campaign_table.csv"))

print("Transaction Data:", transactions.shape)
print("Product Data:", products.shape)
print("Demographic Data:", demographics.shape)
print("Campaign Desc:", campaign_desc.shape)
print("Campaign Table:", campaign_table.shape)

print("\nFirst 5 rows of Transaction Data:")
print(transactions.head())