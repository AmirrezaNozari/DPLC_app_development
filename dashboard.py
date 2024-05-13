from streamlit.components.v1 import html
import streamlit as st
import pydeck as pdk
import pandas as pd
import geocoder
import numpy as np
from sklearn.linear_model import LinearRegression

# # Hide the Deploy button using CSS
# st.markdown("""
# <style>
# #MainMenu {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)

# Title of the app
st.title('4 Lens For different Countries Dashboard')

# Add some text to the app
st.write('4 Lens For different Countries')


# Function to calculate diversity score
def calculate_diversity_score(row_bio):
    score = -187.3967 - 1.3534 * row_bio['Agricultural land (% of land area)'] + \
            10.9765 * row_bio['Forest area (% of land area)'] - \
            32.0721 * row_bio['Wheat Yield (tonnes/km2)']
    return score


def calculate_sum_abs(row_sum):
    return (abs(row_sum['Carbon_Impact']) +
            abs(row_sum['Environment_Impact']) +
            abs(row_sum['Society_Impact']) +
            abs(row_sum['Health_Impact']))


def transform_output(prediction):
    return np.clip(prediction, -100, 100)


def calculate_color(value):
    # Normalize value to the range [0, 1]
    normalized_value = (value + 100) / 200
    # Interpolate color between red and green
    red = int(255 * (1 - normalized_value))
    green = int(255 * normalized_value)
    return f'rgb({red}, {green}, 0)'


# Build model to predict 4lens from Year
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
# Extract years and corresponding carbon impact values
years = np.array(list(carbon_impact.keys())).reshape(-1, 1)
carbon_values = np.array(list(carbon_impact.values()))
# Create and fit the linear regression model
carbon_impact_model = LinearRegression()
carbon_impact_model.fit(years, carbon_values)
max_abs_carbon_value = max(abs(carbon_values.min()), abs(carbon_values.max()))

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
# Extract years and corresponding carbon impact values
years = np.array(list(environment_impact.keys())).reshape(-1, 1)
environment_values = np.array(list(environment_impact.values()))
# Create and fit the linear regression model
environment_impact_model = LinearRegression()
environment_impact_model.fit(years, environment_values)
max_abs_environment_values = max(abs(environment_values.min()), abs(environment_values.max()))

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
# Extract years and corresponding carbon impact values
years = np.array(list(society_impact.keys())).reshape(-1, 1)
society_values = np.array(list(society_impact.values()))
# Create and fit the linear regression model
society_impact_model = LinearRegression()
society_impact_model.fit(years, society_values)
max_abs_society_values = max(abs(society_values.min()), abs(society_values.max()))

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
# Extract years and corresponding carbon impact values
years = np.array(list(health_impact.keys())).reshape(-1, 1)
health_values = np.array(list(health_impact.values()))
# Create and fit the linear regression model
health_impact_model = LinearRegression()
health_impact_model.fit(years, health_values)
max_abs_health_values = max(abs(health_values.min()), abs(health_values.max()))

# Build a model to predict 4lens from biodiversity score #
bio_carbon_impact = {
    24.73: -21.82,
    23.95: -21.07,
    27: -29.35,
    17.32: -21.54,
    20.45: -34.79,
    18.03: -34.47,
    20.08: -37.56,
    11.92: -11.13,
    12.41: -24.41,
    14.02: -15.60,
    7.29: 36,
    9: -6.34,
    3.75: 30.28,
    1.16: 26.99,
    0.82: 20.27,
    -2.43: 32.19,
    -7.26: 42.79,
    -3.9: 30.16,
    -8.15: 28.09,
    -6.31: 22.58
}
# Extract years and corresponding carbon impact values
bio_years = np.array(list(bio_carbon_impact.keys())).reshape(-1, 1)
bio_carbon_values = np.array(list(bio_carbon_impact.values()))
# Create and fit the linear regression model
bio_carbon_impact_model = LinearRegression()
bio_carbon_impact_model.fit(bio_years, bio_carbon_values)
max_abs_bio_carbon_values = max(abs(bio_carbon_values.min()), abs(bio_carbon_values.max()))

bio_environment_impact = {
    24.73: 26.57,
    23.95: 24.12,
    27: 22.41,
    17.32: 39.37,
    20.45: 32.75,
    18.03: 32.68,
    20.08: 21.2,
    11.92: 28.37,
    12.41: 26.45,
    14.02: 4.44,
    7.29: -11.68,
    9: -19.03,
    3.75: -25.91,
    1.16: -22.78,
    -0.82: -20.76,
    -2.43: -27.33,
    -7.26: -31.54,
    -3.9: -34.4,
    -8.15: -27.88,
    -6.31: -29.3
}
# Extract years and corresponding carbon impact values
bio_years = np.array(list(bio_environment_impact.keys())).reshape(-1, 1)
bio_environment_values = np.array(list(bio_environment_impact.values()))
# Create and fit the linear regression model
bio_environment_impact_model = LinearRegression()
bio_environment_impact_model.fit(bio_years, bio_environment_values)
max_abs_bio_environment_values = max(abs(bio_environment_values.min()), abs(bio_environment_values.max()))

bio_society_impact = {
    24.73: 40,
    23.95: 36.1,
    27: 28.82,
    17.32: 6.49,
    20.45: 13.89,
    18.03: 19.51,
    20.08: 6.8,
    11.92: -17.35,
    12.41: -40.67,
    14.01: -14.8,
    7.29: 19.27,
    9: -20.19,
    3.75: -33.27,
    1.16: -19.83,
    0.82: -32.53,
    -2.43: -39.71,
    -7.26: -10.09,
    -3.9: -19.91,
    -8.15: -11.71,
    -6.31: -13.39
}
# Extract years and corresponding carbon impact values
bio_years = np.array(list(bio_society_impact.keys())).reshape(-1, 1)
bio_society_values = np.array(list(bio_society_impact.values()))
# Create and fit the linear regression model
bio_society_impact_model = LinearRegression()
bio_society_impact_model.fit(bio_years, bio_society_values)
max_abs_bio_society_values = max(abs(bio_society_values.min()), abs(bio_society_values.max()))

bio_health_impact = {
    24.73: 11.61,
    23.95: 18.71,
    27: 19.43,
    17.32: 32.59,
    20.45: 18.56,
    18.03: 13.34,
    20.08: 34.44,
    11.92: 43.15,
    12.41: -8.47,
    14.02: -65.16,
    7.29: 33.05,
    9: 54.44,
    3.75: -10.54,
    1.16: -30.41,
    0.82: 26.44,
    -2.43: -0.78,
    -7.26: -15.58,
    -3.9: -15.54,
    -8.15: -32.31,
    -6.31: -34.31
}
# Extract years and corresponding carbon impact values
bio_years = np.array(list(bio_health_impact.keys())).reshape(-1, 1)
bio_health_values = np.array(list(bio_health_impact.values()))
# Create and fit the linear regression model
bio_health_impact_model = LinearRegression()
bio_health_impact_model.fit(bio_years, bio_health_values)
max_abs_bio_health_values = max(abs(bio_health_values.min()), abs(bio_health_values.max()))


# Predict the carbon impact for a given year
def predict_carbon_impact(year):
    prediction = carbon_impact_model.predict(np.array([[year]]))[0]
    scaled_prediction = (prediction / max_abs_carbon_value) * 100
    return scaled_prediction


# Predict the carbon impact for a given year
def predict_environment_impact(year):
    prediction = environment_impact_model.predict(np.array([[year]]))[0]
    scaled_prediction = (prediction / max_abs_environment_values) * 100
    return scaled_prediction


# Predict the carbon impact for a given year
def predict_society_impact(year):
    prediction = society_impact_model.predict(np.array([[year]]))[0]
    scaled_prediction = (prediction / max_abs_society_values) * 100
    return scaled_prediction


# Predict the carbon impact for a given year
def predict_health_impact(year):
    prediction = health_impact_model.predict(np.array([[year]]))[0]
    scaled_prediction = (prediction / max_abs_health_values) * 100
    return scaled_prediction


# Predict the carbon impact for a given year
def predict_bio_carbon_impact(bio_score):
    prediction = bio_carbon_impact_model.predict(np.array([[bio_score]]))[0]
    scaled_prediction = (prediction / max_abs_bio_carbon_values) * 100
    return scaled_prediction


# Predict the carbon impact for a given year
def predict_bio_environment_impact(bio_score):
    prediction = bio_environment_impact_model.predict(np.array([[bio_score]]))[0]
    scaled_prediction = (prediction / max_abs_bio_environment_values) * 100
    return scaled_prediction


# Predict the carbon impact for a given year
def predict_bio_society_impact(bio_score):
    prediction = bio_society_impact_model.predict(np.array([[bio_score]]))[0]
    scaled_prediction = (prediction / max_abs_bio_society_values) * 100
    return scaled_prediction


# Predict the carbon impact for a given year
def predict_bio_health_impact(bio_score):
    prediction = bio_health_impact_model.predict(np.array([[bio_score]]))[0]
    scaled_prediction = (prediction / max_abs_bio_health_values) * 100
    return scaled_prediction


# Load the CSV files
merged_df = pd.read_csv('merged_df.csv')
merged_df['Diversity_Score'] = merged_df.apply(calculate_diversity_score, axis=1)

merged_df['Carbon_Impact'] = merged_df['Year'].map(carbon_impact)
merged_df['Environment_Impact'] = merged_df['Year'].map(environment_impact)
merged_df['Society_Impact'] = merged_df['Year'].map(society_impact)
merged_df['Health_Impact'] = merged_df['Year'].map(health_impact)

merged_df['Sum_ABS'] = merged_df.apply(calculate_sum_abs, axis=1)

merged_df.to_csv('merged_df.csv', index=False)
merged_df = pd.read_csv('merged_df.csv')


def calculate_four_lens(row_lens):
    bio_score = calculate_diversity_score(row_lens)
    lens_carbon_row = predict_bio_carbon_impact(bio_score)
    lens_environment_row = predict_bio_environment_impact(bio_score)
    lens_society_row = predict_bio_society_impact(bio_score)
    lens_health_row = predict_bio_health_impact(bio_score)

    return lens_carbon_row, lens_environment_row, lens_society_row, lens_health_row


years = merged_df['Year'].unique()

# Sidebar for selecting existing data (year and country)
st.sidebar.subheader("Select Existing Data")
# Dropdown for selecting year
selected_year = st.sidebar.selectbox('Select Year', years)

# Dropdown for selecting country
selected_country = st.sidebar.selectbox('Select Country', merged_df['Country Name'].unique())

selected_agricultural = 0
selected_forest = 0
selected_wheat = 0

# Add a sidebar
st.sidebar.header('Choose Data')

# Checkbox for showing user input section
show_user_input = st.sidebar.checkbox('Enter User Input', value=True)

# Sidebar for entering user input (conditionally displayed)
if show_user_input:
    st.sidebar.subheader("Enter User Input")
    # Text input for agricultural land
    selected_agricultural = st.sidebar.text_input('Enter Agricultural land (% of land area):', '')

    # Text input for forest area
    selected_forest = st.sidebar.text_input('Enter Forest area (% of land area):', '')

    # Text input for wheat yield
    selected_wheat = st.sidebar.text_input('Enter Wheat Yield (tonnes/km2):', '')

# Filter data based on selected country
country_data = merged_df[merged_df['Country Name'] == selected_country]
#
# # Display selected data
# if selected_agricultural == 'Agricultural land':
#     selected_column = 'Agricultural land (% of land area)'
# elif selected_data == 'Forest area':
#     selected_column = 'Forest area (% of land area)'
# else:
#     selected_column = 'Wheat Yield (tonnes/km2)'

# Calculate four lens based on user input
if selected_agricultural and selected_forest and selected_wheat:
    lens_carbon, lens_environment, lens_society, lens_health = calculate_four_lens({
        'Agricultural land (% of land area)': float(selected_agricultural),
        'Forest area (% of land area)': float(selected_forest),
        'Wheat Yield (tonnes/km2)': float(selected_wheat)
    })
    user_diversity_score = calculate_diversity_score({
        'Agricultural land (% of land area)': float(selected_agricultural),
        'Forest area (% of land area)': float(selected_forest),
        'Wheat Yield (tonnes/km2)': float(selected_wheat)
    })
    lens_values = [lens_carbon, lens_environment, lens_society, lens_health]
    print("# 1 #")
    # print(lens_values)
    normalized_lens_values = [(value - min(lens_values)) / (max(lens_values) - min(lens_values)) * 200 - 100 for value
                              in lens_values]
    # print(normalized_lens_values)
    sum_normalized = sum(normalized_lens_values)
    if sum_normalized == 0:
        adjusted_lens_values = normalized_lens_values
    else:
        scale_factor = 100 / sum_normalized
        adjusted_lens_values = [value * scale_factor for value in normalized_lens_values]
    adjusted_lens_values = np.clip(adjusted_lens_values, -100, 100)
    # print(adjusted_lens_values)
    user_lens_df = pd.DataFrame({
        'Lens': ['Carbon', 'Environment', 'Society', 'Health'],
        'Value': adjusted_lens_values  # [lens_carbon, lens_environment, lens_society, lens_health]
    })

    # Display scorecard for user input
    # if selected_agricultural and selected_forest and selected_wheat:
    #     st.subheader('User Input Scorecard')
    #     st.write(user_lens_df)
    #
    #     st.write('User Input Biodiversity Score:', user_diversity_score)
    # Display lenses and biodiversity score as circles in the same row
    st.subheader('User Input Scorecard')

    # Define CSS style for the container
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

    # Begin the circle container
    circle_html = """
    <div class='circle-container'>
    """
    print("*******")
    print(user_lens_df.head())
    print("*******")
    # Display lenses as circles with lens number on top
    for i, row in user_lens_df.iterrows():
        lens_number = row['Lens']
        lens_value = float(row['Value'])
        background_color = calculate_color(lens_value)
        circle_html += f"""
        <div class='circle' style='background-color: {background_color}'>
            <div class='circle-number' style='font-size: 10px;'>{lens_number} Impact(%)</div>
            {lens_value:.2f}  <!-- Format as float -->
        </div>
        """

    # Display biodiversity score in a green circle
    circle_html += f"""
    <div class='circle' style='background-color: #1f77b4;'>
        <div class='circle-number' style='font-size: 10px;'>Biodiversity</div>
        {user_diversity_score:.2f}
    </div>
    """

    # End the circle container
    circle_html += """
    </div>
    """

    # Render HTML using components.html
    html_output = css_style + circle_html
    html(html_output)

# Display scorecard for merged data
st.subheader('History Data')
st.write('Select a country from the sidebar to see scorecard based on existing data.')

# Display scorecard based on merged data
if selected_country:
    lens_carbon, lens_environment, lens_society, lens_health = calculate_four_lens(country_data.iloc[0])
    # country_lens = list(country_lens)
    country_diversity_score = country_data.iloc[0]['Diversity_Score']
    lens_values = [lens_carbon, lens_environment, lens_society, lens_health]
    print("# 2 #")
    # print(lens_values)
    abs_max = max(map(abs, lens_values))
    normalized_lens_values = [value / abs_max * 100 for value in lens_values]
    # print(normalized_lens_values)
    sum_normalized = sum(normalized_lens_values)
    if sum_normalized == 0:
        adjusted_lens_values = normalized_lens_values
    else:
        scale_factor = 100 / sum_normalized
        adjusted_lens_values = [value * scale_factor for value in normalized_lens_values]
    adjusted_lens_values = np.clip(adjusted_lens_values, -100, 100)
    # print(adjusted_lens_values)
    country_lens_df = pd.DataFrame({
        'Lens': ['Carbon', 'Environment', 'Society', 'Health'],
        'Value': adjusted_lens_values  # [lens_carbon, lens_environment, lens_society, lens_health]
    })

    # Display lenses and biodiversity score as circles in the same row
    st.subheader('Visualizations')

    # Define CSS style for the container
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

    # Begin the circle container
    circle_html = """
    <div class='circle-container'>
    """

    # Display lenses as circles with lens number on top
    for i, row in country_lens_df.iterrows():
        lens_number = row['Lens']
        lens_value = float(row['Value'])
        background_color = calculate_color(lens_value)
        circle_html += f"""
        <div class='circle' style='background-color: {background_color}'>
            <div class='circle-number' style='font-size: 10px;'>{lens_number} Impact(%)</div>
            {lens_value:.2f}
        </div>
        """

    # Display biodiversity score in a green circle
    circle_html += f"""
    <div class='circle' style='background-color: #1f77b4;'>
        <div class='circle-number' style='font-size: 10px;'>Biodiversity</div>
        {country_diversity_score:.2f}
    </div>
    """

    # End the circle container
    circle_html += """
    </div>
    """

    # Render HTML using components.html
    html_output = css_style + circle_html
    html(html_output)

# Calculate diversity score
# country_data['Diversity Score'] = country_data.apply(calculate_diversity_score, axis=1)

# Create a PyDeck scatter plot layer
# layer = pdk.Layer(
#     "ScatterplotLayer",
#     data=country_data,
#     get_position=["Longitude", "Latitude"],
#     get_fill_color=[255, 0, 0, 200],  # RGBA color for the points
#     get_radius=100000,
# )
#
# # Set the viewport for the PyDeck map
# view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)
#
# # Create a PyDeck map
# map_1 = pdk.Deck(
#     map_style="mapbox://styles/mapbox/light-v9",
#     layers=[layer],
#     initial_view_state=view_state,
# )

# Define the tooltip content
tooltip_content = {
    "html": "<b>Biodiversity Score:</b> {Diversity_Score}",
    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}

map_styles = {
    "Light": "mapbox://styles/mapbox/light-v9",
    "Dark": "mapbox://styles/mapbox/dark-v9",
    "Satellite": "mapbox://styles/mapbox/satellite-v9"
}

selected_style = st.sidebar.selectbox("Select Map Style", list(map_styles.keys()), index=0)

# Create a PyDeck map with tooltip and zoom focused on the selected country
map_1 = pdk.Deck(
    map_style=map_styles[selected_style],
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=country_data,
            get_position=["Longitude", "Latitude"],
            get_fill_color=[255, 0, 0, 50],  # RGBA color for the points
            get_radius=100,
            tooltip=tooltip_content,  # Add tooltip
        )
    ],
    initial_view_state={
        "latitude": country_data["Latitude"].mean(),  # Center latitude of the selected country
        "longitude": country_data["Longitude"].mean(),  # Center longitude of the selected country
        "zoom": 3  # Adjust zoom level as needed
    }
)

# Render the PyDeck map in the Streamlit app
st.pydeck_chart(map_1)

