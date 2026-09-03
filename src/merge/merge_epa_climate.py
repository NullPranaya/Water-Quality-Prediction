import pandas as pd
import os

def merge_epa_climate():
    print("Loading data...")
    # Load EPA merged data
    epa_df = pd.read_csv('data/tabular/merged/epa-merged.csv')
    
    # Load Climate data
    climate_df = pd.read_csv('data/tabular/climate/clean/isu-climate-clean.csv')
    
    # Load Mapping data
    mapping_df = pd.read_csv('data/tabular/merged/epa-to-climate-station-map.csv')
    
    print("Preparing EPA data for merge...")
    # 1. Join EPA data with the mapping to get the climate station for each EPA location
    # Only keep the necessary columns from mapping
    mapping_subset = mapping_df[['MonitoringLocationIdentifier', 'climate_station', 'climate_station_name', 'distance_to_climate_station_km']]
    epa_mapped = epa_df.merge(mapping_subset, on='MonitoringLocationIdentifier', how='left')
    
    # 2. Extract date from ActivityStartDateTime to merge with climate 'day'
    # ActivityStartDateTime is in format 'YYYY-MM-DD HH:MM:SS'
    epa_mapped['day'] = pd.to_datetime(epa_mapped['ActivityStartDateTime']).dt.strftime('%Y-%m-%d')
    
    print("Merging with climate data...")
    # 3. Merge with climate data on (climate_station, day)
    # Climate data 'day' column is also in 'YYYY-MM-DD' format based on preview
    merged_df = epa_mapped.merge(
        climate_df, 
        left_on=['climate_station', 'day'], 
        right_on=['station', 'day'], 
        how='left'
    )
    
    # Clean up redundant columns if any
    if 'station' in merged_df.columns:
        merged_df = merged_df.drop(columns=['station'])
    
    print(f"Merge complete. Merged shape: {merged_df.shape}")
    
    # Save the result
    output_path = 'data/tabular/merged/epa-climate-merged.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

if __name__ == "__main__":
    merge_epa_climate()
