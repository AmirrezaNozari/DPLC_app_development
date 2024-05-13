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

if show_user_input:
    st.sidebar.subheader("Enter User Input")

    selected_agricultural = st.sidebar.text_input('Enter Agricultural land (% of land area):', '')
    selected_forest = st.sidebar.text_input('Enter Forest area (% of land area):', '')
    selected_wheat = st.sidebar.text_input('Enter Wheat Yield (tonnes/km2):', '')

country_data = merged_df[merged_df['Country Name'] == selected_country]
#
# if selected_agricultural == 'Agricultural land':
#     selected_column = 'Agricultural land (% of land area)'
# elif selected_data == 'Forest area':
#     selected_column = 'Forest area (% of land area)'
# else:
#     selected_column = 'Wheat Yield (tonnes/km2)'

if selected_agricultural and selected_forest and selected_wheat:
    user_lens = calculate_four_lens({
        'Agricultural land (% of land area)': float(selected_agricultural),
        'Forest area (% of land area)': float(selected_forest),
        'Wheat Yield (tonnes/km2)': float(selected_wheat)
    })
    user_diversity_score = calculate_diversity_score({
        'Agricultural land (% of land area)': float(selected_agricultural),
        'Forest area (% of land area)': float(selected_forest),
        'Wheat Yield (tonnes/km2)': float(selected_wheat)
    })
    user_lens_df = pd.DataFrame({
        'Lens': ['Lens 1', 'Lens 2', 'Lens 3', 'Lens 4'],
        'Value': user_lens
    })

    # if selected_agricultural and selected_forest and selected_wheat:
    #     st.subheader('User Input Scorecard')
    #     st.write(user_lens_df)
    #
    #     st.write('User Input Biodiversity Score:', user_diversity_score)
    st.subheader('User Input Scorecard')

    st.write("")

    css_style = """
    <style>
    .circle-container {
        display: flex;
        margin-top: 30px; /* Adjust the margin-top value as needed */
    }

    .circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background-color: #1f77b4;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 20px;
        position: relative;
    }

    .circle-number {
        position: absolute;
        top: -25px;
        left: 0;
        right: 0;
        text-align: center;
        color: black;
    }
    </style>
    """

    circle_html = """
    <div class='circle-container'>
    """
    print("*******")
    print(user_lens_df.head())
    print("*******")
    for i, row in user_lens_df.iterrows():
        lens_number = row['Lens'].split()[1]
        lens_value = float(row['Value'])
        circle_html += f"""
        <div class='circle'>
            <div class='circle-number'>Lens {lens_number}</div>
            {lens_value:.2f}  <!-- Format as float -->
        </div>
        """

    circle_html += f"""
    <div class='circle' style='background-color: #2ca02c;'>
        <div class='circle-number'>Biodiversity</div>
        {user_diversity_score:.2f}
    </div>
    """

    circle_html += """
    </div>
    """

    html_output = css_style + circle_html
    html(html_output)

st.subheader('Merged Data Scorecard')
st.write('Select a country from the sidebar to see scorecard based on merged data.')

if selected_country:
    country_lens = calculate_four_lens(country_data.iloc[0])
    country_lens = list(country_lens)
    country_diversity_score = country_data.iloc[0]['Diversity Score']
    country_lens_df = pd.DataFrame({
        'Lens': ['Lens 1', 'Lens 2', 'Lens 3', 'Lens 4'],
        'Value': country_lens
    })

    st.subheader('Visualizations')

    css_style = """
    <style>
    .circle-container {
        display: flex;
        margin-top: 30px; /* Adjust the margin-top value as needed */
    }

    .circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background-color: #1f77b4;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 20px;
        position: relative;
    }

    .circle-number {
        position: absolute;
        top: -25px;
        left: 0;
        right: 0;
        text-align: center;
        color: black;
    }
    </style>
    """

    circle_html = """
    <div class='circle-container'>
    """

    for i, lens_value in enumerate(country_lens):
        lens_number = i + 1
        circle_html += f"""
        <div class='circle'>
            <div class='circle-number'>Lens {lens_number}</div>
            {lens_value:.2f}
        </div>
        """

    circle_html += f"""
    <div class='circle' style='background-color: #2ca02c;'>
        <div class='circle-number'>Biodiversity</div>
        {country_diversity_score:.2f}
    </div>
    """

    circle_html += """
    </div>
    """

    html_output = css_style + circle_html
    html(html_output)

country_data['Diversity Score'] = country_data.apply(calculate_diversity_score, axis=1)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=country_data,
    get_position=["Longitude", "Latitude"],
    get_fill_color=[255, 0, 0, 200],  # RGBA color for the points
    get_radius=100000,
)

view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)

map_1 = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    layers=[layer],
    initial_view_state=view_state,
)

st.pydeck_chart(map_1)

# st.write('## Diversity Score Card')
# st.write(f'Country: {selected_country}')
# st.write(f'Selected Data: {selected_data}')
# st.write(f'Diversity Score: {country_data["Diversity Score"].values[0]}')