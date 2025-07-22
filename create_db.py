import pandas as pd
import sqlite3

conn = sqlite3.connect("ecommerce.db")

# 1. Load Ad Sales data
df_ad = pd.read_csv("Product-Level Ad Sales and Metrics (mapped).csv")
df_ad.to_sql("ad_sales", conn, if_exists="replace", index=False)
print("✅ ad_sales table created")

# 2. Load Total Sales data
df_total = pd.read_csv("Product-Level Total Sales and Metrics (mapped).csv")
df_total.to_sql("total_sales", conn, if_exists="replace", index=False)
print("✅ total_sales table created")

# 3. Load Eligibility data
df_elig = pd.read_csv("Product-Level Eligibility Table (mapped).csv")
df_elig.to_sql("eligibility", conn, if_exists="replace", index=False)
print("✅ eligibility table created")

conn.close()
print(" ecommerce.db created successfully!")
