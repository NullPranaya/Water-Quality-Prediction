import pandas as pd
import numpy as np

def merge_epa_climate_agriculture():
    print("Loading data...")
    epa_climate_df = pd.read_csv('data/tabular/merged/epa-climate-merged.csv')
    ag_df = pd.read_csv('data/tabular/agricultural/clean/usdaNass-agriculture-clean.csv')
    
    print("Preparing data for merge...")
    epa_climate_df['day_dt'] = pd.to_datetime(epa_climate_df['day'])
    epa_climate_df['year'] = epa_climate_df['day_dt'].dt.year
    epa_climate_df['state'] = 'IOWA' 
    # Pivot ag data - keep year, period, state as index
    ag_df['data_item_short'] = ag_df['data_item'].str.replace(', MEASURED IN .*', '', regex=True)
    ag_pivot = ag_df.pivot_table(
        index=['year', 'period', 'state'],
        columns='data_item_short',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Create two versions of ag_pivot: one for months and one for 'YEAR'
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    ag_month = ag_pivot[ag_pivot['period'].isin(months)].copy()
    ag_year = ag_pivot[ag_pivot['period'] == 'YEAR'].copy()
    
    # Prefix columns to distinguish them
    ag_month_cols = {c: f"AG_MONTH_{c}" for c in ag_month.columns if c not in ['year', 'period', 'state']}
    ag_month.rename(columns=ag_month_cols, inplace=True)
    ag_month.rename(columns={'period': 'month'}, inplace=True)
    
    ag_year_cols = {c: f"AG_YEAR_{c}" for c in ag_year.columns if c not in ['year', 'period', 'state']}
    ag_year.rename(columns=ag_year_cols, inplace=True)
    ag_year.drop(columns=['period'], inplace=True)
    
    # Prepare EPA data
    epa_climate_df['month'] = epa_climate_df['day_dt'].dt.strftime('%b').str.upper()
    
    print("Merging with Yearly Agriculture data...")
    merged_df = epa_climate_df.merge(
        ag_year,
        on=['year', 'state'],
        how='left'
    )
    
    print("Merging with Monthly Agriculture data...")
    merged_df = merged_df.merge(
        ag_month,
        left_on=['year', 'month', 'state'],
        right_on=['year', 'month', 'state'],
        how='left'
    )
    
    print(f"Final merged shape: {merged_df.shape}")
    
    output_path = 'data/tabular/merged/epa-climate-agriculture-merged.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

if __name__ == "__main__":
    merge_epa_climate_agriculture()
