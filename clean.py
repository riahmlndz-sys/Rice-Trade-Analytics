import pandas as pd

df = pd.read_csv('clean_rice_data.csv')

df = df[
    (df['Area'] == 'Philippines') &
    (df['Item'].str.lower().str.contains('rice, milled'))
]

df = df[['Year', 'Element', 'Value']]

df_pivot = df.pivot(index='Year', columns='Element', values='Value').reset_index()

df_pivot.columns.name = None
df_pivot.columns = ['year'] + [col.strip().lower().replace(' ', '_') for col in df_pivot.columns[1:]]

df_pivot = df_pivot.sort_values(by='year')
df_pivot = df_pivot.dropna()

df_pivot.to_csv('cleaned_rice_milled_trade_data.csv', index=False)

print("Cleaned data preview:")
print(df_pivot.head())

import os

output_path = os.path.abspath("cleaned_rice_milled_trade_data.csv")
print(f"Saved file to: {output_path}")