import pandas as pd
import simfin as sf

# 1. Set up your SimFin credentials
sf.set_api_key('use ur own api key')
sf.set_data_dir('~/simfin_data/') 

print("Downloading and loading bulk data matrices...")
df_income = sf.load_income(variant='quarterly', market='us')
df_balance = sf.load_balance(variant='quarterly', market='us')
# Switch to 'daily' so we have historical prices to match the reports
df_prices = sf.load_shareprices(variant='daily', market='us') 

tickers_to_track = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
    "XOM", "CVX", "CAT", "GE", "MMM", "WMT", "COST", "TGT", "KO", "PEP", 
    "NKE", "DIS", "JNJ", "PFE", "MRK", "UNH", "LLY", "F", "GM", "BA",
    "ORCL", "CSCO", "IBM", "HD", "MCD", "ADBE", "CRM", "TXN", "QCOM", "AMAT"
]

print("Filtering and merging datasets into timeline...")
df_income = df_income.reset_index()
df_balance = df_balance.reset_index()
df_prices = df_prices.reset_index()

# Safely filter
df_income = df_income[df_income['Ticker'].isin(tickers_to_track)]
df_balance = df_balance[df_balance['Ticker'].isin(tickers_to_track)]
df_prices = df_prices[df_prices['Ticker'].isin(tickers_to_track)]

# Merge Income and Balance Sheet exactly on the Report Date
df_fundamentals = pd.merge(df_income, df_balance, on=['Ticker', 'Report Date'], how='inner')

# Sort by dates (required for merge_asof)
df_fundamentals = df_fundamentals.sort_values('Report Date')
df_prices = df_prices.sort_values('Date')

# Snap the closest trading day 'Close' price to the 'Report Date'
df_merged = pd.merge_asof(
    df_fundamentals, 
    df_prices[['Ticker', 'Date', 'Close']],
    left_on='Report Date', 
    right_on='Date', 
    by='Ticker', 
    direction='nearest'
)

# Extract final columns including the new 'Close' price
final_dataset = df_merged[[
    'Ticker', 
    'Report Date',
    'Close', 
    'Net Income', 
    'Total Liabilities', 
    'Total Equity', 
    'Total Current Assets', 
    'Total Current Liabilities'
]].copy()

# Rename to match the ML pipeline schema
final_dataset.columns = ['ticker', 'report_date', 'price', 'net_income', 'total_debt', 'equity', 'current_assets', 'current_liabilities']

# Scale everything except price to Millions
for col in ['net_income', 'total_debt', 'equity', 'current_assets', 'current_liabilities']:
    final_dataset[col] = final_dataset[col] / 1e6

final_dataset.to_csv("historical_fundamentals.csv", index=False)
print(f"Success! Saved {len(final_dataset)} historical data nodes to 'historical_fundamentals.csv'.")