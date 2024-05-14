import pandas as pd
import geocoder

agricultural_df = pd.read_csv('3_Outlier Agricultural Land.csv')
forest_df = pd.read_csv('3_Outlier Forest Area.csv')
wheat_yield_df = pd.read_csv('3_Outlier wheat-yields.csv')

merged_df = pd.merge(agricultural_df, forest_df, on=['Country Name', 'Year'])
merged_df = pd.merge(merged_df, wheat_yield_df, on=['Country Name', 'Year'])

unique_countries = merged_df['Country Name'].unique()

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

merged_df = pd.merge(merged_df, country_lat_lon_df, on='Country Name', how='left')

merged_df.dropna(subset=['Country Name', 'Year', 'Latitude', 'Longitude'], inplace=True)
merged_df.drop_duplicates(inplace=True)
merged_df.to_csv('merged_df.csv', index=False)
