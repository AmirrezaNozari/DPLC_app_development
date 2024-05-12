import pandas as pd
import geocoder

# Load the CSV files
agricultural_df = pd.read_csv('3_Outlier Agricultural Land.csv')
forest_df = pd.read_csv('3_Outlier Forest Area.csv')
wheat_yield_df = pd.read_csv('3_Outlier wheat-yields.csv')

# Merge data frames on 'Country Name' and 'Year'
merged_df = pd.merge(agricultural_df, forest_df, on=['Country Name', 'Year'])
merged_df = pd.merge(merged_df, wheat_yield_df, on=['Country Name', 'Year'])

# Create a list of unique countries
unique_countries = merged_df['Country Name'].unique()

# Create a DataFrame to store country latitude and longitude data
country_lat_lon_df = pd.DataFrame(columns=['Country Name', 'Latitude', 'Longitude'])

# https://www.bingmapsportal.com/Application#
for country in unique_countries:
    g = geocoder.bing(country, key='Agu0cqK7UvXzLOD19vvSFlsmoayaeaDaWVgX7YUaDKg60nxFfKPqJc4wQJrYEQsO')
    results = g.json
    if results:
        country_lat_lon_df = pd.concat([country_lat_lon_df,
                                        pd.DataFrame([[country, results['lat'], results['lng']]],
                                                     columns=['Country Name', 'Latitude',
                                                              'Longitude'])], ignore_index=True)

# Merge country latitude and longitude data with merged_df
merged_df = pd.merge(merged_df, country_lat_lon_df, on='Country Name', how='left')

# Drop rows with missing latitude or longitude values
merged_df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

merged_df.to_csv('merged_df.csv', index=False)
