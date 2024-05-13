from streamlit.components.v1 import html
import streamlit as st
import pydeck as pdk
import pandas as pd
import geocoder
import numpy as np
from sklearn.linear_model import LinearRegression

st.title('4 Lens For different Countries Dashboard')
st.write('4 Lens For different Countries')


def calculate_diversity_score(row):
    score = -187.3967 - 1.3534 * row['Agricultural land (% of land area)'] + \
            10.9765 * row['Forest area (% of land area)'] - \
            32.0721 * row['Wheat Yield (tonnes/km2)']
    return score


def calculate_carbon_impact(row):
    return 2


def calculate_environment_impact(row):
    return 2


def calculate_society_impact(row):
    return 2


def calculate_health_impact(row):
    return 2


def calculate_sum_abs(row):
    return (abs(row['Carbon_Impact']) +
            abs(row['Environment_Impact']) +
            abs(row['Society_Impact']) +
            abs(row['Health_Impact']))


carbon_impact = {
    2001: -21.82,
    2002: -21.07,
    2003: -29.35,
    2004: -21.54,
    2005: -34.79,
    2006: -34.47,
    2007: -37.56,
    2008: -11.13,
    2009: -24.41,
    2010: -15.60,
    2011: 36,
    2012: -6.34,
    2013: 30.28,
    2014: 26.99,
    2015: 20.27,
    2016: 32.19,
    2017: 42.79,
    2018: 30.16,
    2019: 28.09,
    2020: 22.58
}

years = np.array(list(carbon_impact.keys())).reshape(-1, 1)
carbon_values = np.array(list(carbon_impact.values()))

carbon_impact_model = LinearRegression()
carbon_impact_model.fit(years, carbon_values)

environment_impact = {
    2001: 26.57,
    2002: 24.12,
    2003: 22.41,
    2004: 39.37,
    2005: 32.75,
    2006: 32.68,
    2007: 21.2,
    2008: 28.37,
    2009: 26.45,
    2010: 4.44,
    2011: -11.68,
    2012: -19.03,
    2013: -25.91,
    2014: -22.78,
    2015: -20.76,
    2016: -27.33,
    2017: -31.54,
    2018: -34.4,
    2019: -27.88,
    2020: -29.3
}

years = np.array(list(environment_impact.keys())).reshape(-1, 1)
environment_values = np.array(list(environment_impact.values()))

environment_impact_model = LinearRegression()
environment_impact_model.fit(years, environment_values)

society_impact = {
    2001: 40,
    2002: 36.1,
    2003: 28.82,
    2004: 6.49,
    2005: 13.89,
    2006: 19.51,
    2007: 6.8,
    2008: -17.35,
    2009: -40.67,
    2010: -14.8,
    2011: 19.27,
    2012: -20.19,
    2013: -33.27,
    2014: -19.83,
    2015: -32.53,
    2016: -39.71,
    2017: -10.09,
    2018: -19.91,
    2019: -11.71,
    2020: -13.39
}

years = np.array(list(society_impact.keys())).reshape(-1, 1)
society_values = np.array(list(society_impact.values()))

society_impact_model = LinearRegression()
society_impact_model.fit(years, society_values)

health_impact = {
    2001: 11.61,
    2002: 18.71,
    2003: 19.43,
    2004: 32.59,
    2005: 18.56,
    2006: 13.34,
    2007: 34.44,
    2008: 43.15,
    2009: -8.47,
    2010: -65.16,
    2011: 33.05,
    2012: 54.44,
    2013: -10.54,
    2014: -30.41,
    2015: 26.44,
    2016: -0.78,
    2017: -15.58,
    2018: -15.54,
    2019: -32.31,
    2020: -34.31
}

years = np.array(list(health_impact.keys())).reshape(-1, 1)
health_values = np.array(list(health_impact.values()))

health_impact_model = LinearRegression()
health_impact_model.fit(years, health_values)


def predict_carbon_impact(year):
    return carbon_impact_model.predict(np.array([[year]]))[0]


def predict_environment_impact(year):
    return environment_impact_model.predict(np.array([[year]]))[0]


def predict_society_impact(year):
    return society_impact_model.predict(np.array([[year]]))[0]


def predict_health_impact(year):
    return health_impact_model.predict(np.array([[year]]))[0]


merged_df = pd.read_csv('merged_df.csv')
merged_df['Diversity Score'] = merged_df.apply(calculate_diversity_score, axis=1)

merged_df['Carbon_Impact'] = merged_df['Year'].map(carbon_impact)
merged_df['Environment_Impact'] = merged_df['Year'].map(environment_impact)
merged_df['Society_Impact'] = merged_df['Year'].map(society_impact)
merged_df['Health_Impact'] = merged_df['Year'].map(health_impact)

merged_df['Sum_ABS'] = merged_df.apply(calculate_sum_abs, axis=1)

merged_df['Carbon_Impact_Percentage'] = merged_df.apply(calculate_diversity_score, axis=1)
merged_df['Environment_Impact_Percentage'] = merged_df.apply(calculate_diversity_score, axis=1)
merged_df['Society_Impact_Percentage'] = merged_df.apply(calculate_diversity_score, axis=1)
merged_df['Health_Impact_Percentage'] = merged_df.apply(calculate_diversity_score, axis=1)

merged_df.to_csv('merged_df.csv', index=False)
merged_df = pd.read_csv('merged_df.csv')


def calculate_four_lens(row):
    len1 = row['Agricultural land (% of land area)'] * 1.1
    len2 = row['Forest area (% of land area)'] * 1.2
    len3 = row['Wheat Yield (tonnes/km2)'] * 1.3
    len4 = (len1 + len2 + len3) / 3
    return len1, len2, len3, len4


years = merged_df['Year'].unique()

st.sidebar.header('Choose Data')
show_user_input = st.sidebar.checkbox('Enter User Input')
st.sidebar.subheader("Select Existing Data")
selected_year = st.sidebar.selectbox('Select Year', years)
selected_country = st.sidebar.selectbox('Select Country', merged_df['Country Name'].unique())

selected_agricultural = 0
selected_forest = 0

