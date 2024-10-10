import subprocess
import sys

# List of libraries to install
libraries = [
    "plotly",
    "numpy",
    "spacy",
    "networkx",
    "pyvis",
    "streamlit",
    "pandas",
    "textblob",
    "matplotlib"
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for library in libraries:
    try:
        __import__(library)
    except ImportError:
        print(f"{library} not found. Installing...")
        install(library)
    else:
        print(f"{library} is already installed.")


python -m spacy download en_core_web_sm


import plotly.graph_objects as go
import numpy as np
import spacy
import re
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
# from wordcloud import WordCloud
from io import BytesIO
import os
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)

# Files grouped by country and energy type
country_energy_files = {
    "Australia - Nuclear": [
        os.path.join(dirname,"data/aus_public_opinion_1.txt"),
        os.path.join(dirname,"data/aus_public_opinion_2.txt"),
        os.path.join(dirname,"data/aus_public_opinion_3.txt"),
        os.path.join(dirname,"data/aus_public_opinion_4.txt")
    ],
    "Australia - Solar": [
        os.path.join(dirname,"data/aus_public_opinion_5.txt"),
        os.path.join(dirname,"data/aus_public_opinion_6.txt")
    ],
    "France - Nuclear": [
        os.path.join(dirname,"data/france_public_opinion_1.txt"),
        os.path.join(dirname,"data/france_public_opinion_3.txt")
    ],
    "France - Solar": [
        os.path.join(dirname,"data/france_public_opinion_1.txt"),
        os.path.join(dirname,"data/france_public_opinion_2.txt")
    ],
    "Singapore - Nuclear": [
        os.path.join(dirname,"data/singapore_public_opinion_3.txt")
    ],
    "Singapore - Solar": [
        os.path.join(dirname,"data/singapore_public_opinion_2.txt")
    ]
}

# File paths for each country's policy documents
Australia_Policy = [
    os.path.join(dirname,"data/Australia2023EnergyPolicyReview_extracted_paragraphs.txt"),
    os.path.join(dirname,"data/AustraliaPublicPolicy.txt"),
    os.path.join(dirname,"data/Australia’s network of nuclear cooperation agreements.txt")
]

France_Policy = [
    os.path.join(dirname,"data/Energy_Policy_France_2016_Review_extracted_paragraphs.txt")
]

Singapore_Policy = [
    os.path.join(dirname,"data/singapore_nr_extracted_paragraphs.txt"),
    os.path.join(dirname,"data/SingaporePublicPolicy.txt")
]

# Load the datasets for electricity production
data = pd.read_csv(os.path.join(dirname,'data/electricity-production-by-source.csv'))
final_data = pd.read_csv(os.path.join(dirname,'data/electricity-production-with-forecast.csv'))

# Data preparation for electricity production
energy_sources = ['Coal', 'Gas', 'Nuclear', 'Hydro', 'Solar', 'Oil', 'Wind', 'Bioenergy']
long_format = data.melt(id_vars=['Year', 'Entity'], value_vars=energy_sources, var_name='Source',
                        value_name='Production')
selected_countries = ['Australia', 'Singapore', 'France']
long_format = long_format[long_format['Entity'].isin(selected_countries)]
filtered_data = final_data[final_data['Entity'].isin(selected_countries)]

# Load the dataset for solar energy
solar_data = pd.read_csv(os.path.join(dirname,'data/solar-energy-consumption.csv'))
solar_data_filtered = solar_data[(solar_data['Year'] >= 2008) & (solar_data['Code'].isin(['AUS', 'FRA', 'SGP']))]
solar_data_filtered = solar_data_filtered.dropna(subset=['Electricity from solar - TWh'])

# Create a choropleth map
choropleth_map = px.choropleth(
    solar_data_filtered,
    locations="Code",
    color="Electricity from solar - TWh",
    hover_name="Entity",
    animation_frame="Year",
    color_continuous_scale='reds',
    range_color=[0, solar_data_filtered['Electricity from solar - TWh'].max()],
    labels={'Electricity from solar - TWh': 'Solar Generation (TWh)'},
    title="Solar Power Generation: Australia, France, Singapore (2008-2023)"
)

# Create a line chart with trend lines
line_chart = px.line(
    solar_data_filtered,
    x='Year',
    y='Electricity from solar - TWh',
    color='Entity',
    markers=True,
    labels={'Electricity from solar - TWh': 'Solar Power Generation (TWh)'},
    title='Solar Power Generation Trends (2008-2023)'
)

# Fit a global regression line across all countries
coefficients = np.polyfit(solar_data_filtered['Year'], solar_data_filtered['Electricity from solar - TWh'], 1)
x_trend = np.linspace(solar_data_filtered['Year'].min(), solar_data_filtered['Year'].max(), 100)
y_trend = np.polyval(coefficients, x_trend)
line_chart.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name='Global Trend Line',
                                line=dict(color='black', width=4, dash='dot')))

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Keywords dictionary for entity extraction
keywords = {
    'people': [
        'Lenka Kollar', 'Colin Watson', 'Trevor Danos', 'Anna White', 'Emily Andrews', 'Milaan Latten',
        'Michael Gunner', 'Tony Abbott', 'Julie Bishop', 'Andrew Robb', 'Lawrence Wong', 'Olivier Carré',
        'Jonathan Vausort', 'Victorien Erussard', 'Yacine Ait Kaci'
    ],
    'organizations': [
        'SMR Nuclear Technology Pty Ltd', 'Helixos', 'Sun Cable', 'SMEC', 'Bechtel', 'Hatch',
        'Corrs Chambers Westgarth', 'Engineers Australia',
        'Australian Academy of Technological Sciences and Engineering',
        'Australian Radiation Protection and Nuclear Safety Agency (ARPANSA)', 'The Australia Institute',
        'Orano', 'Amarenco', 'French government', 'Energy Market Authority (EMA)',
        'Singapore Alliance with France for Fusion Energy',
        'ASEAN', 'CNR', 'ENGIE', 'Energy Observer', 'Nanyang Technological University',
        'Solar Energy Research Institute of Singapore (SERIS)'
    ],
    'energy_types': [
        'nuclear energy', 'solar energy', 'wind energy', 'renewable energy', 'coal',
        'hydroelectricity', 'small modular reactors (SMRs)', 'battery storage', 'carbon emissions',
        'floating nuclear power plants (FNPPs)', 'geothermal energy', 'biofuels', 'hydropower',
        'Molten Salt Reactors (MSRs)', 'High-Temperature Gas-Cooled Reactors (HTGRs)',
        'Integral Pressurised Water Reactors (iPWR)', 'Fast Reactors', 'Travelling Wave Reactors (TWRs)',
        'Very High-Temperature Reactors (VHTRs)', 'Liquid Metal-cooled Reactors'
    ],
    'countries': [
        'Australia', 'Singapore', 'United States', 'United Kingdom', 'Indonesia', 'China',
        'France', 'Malaysia', 'Thailand', 'Laos', 'Philippines'
    ]
}

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Entity Extraction combining SpaCy and keyword-based extraction
def extract_entities(text):
    doc = nlp(text)
    extracted_entities = defaultdict(list)

    # Extract entities using SpaCy NER
    for ent in doc.ents:
        entity_name = ent.text.strip().lower()
        if ent.label_ in ['PERSON', 'GPE', 'ORG', 'LAW']:
            extracted_entities[ent.label_].append(entity_name)

    # Keyword-based extraction
    for entity_type, keyword_list in keywords.items():
        for keyword in keyword_list:
            if keyword.lower() in text:
                extracted_entities[entity_type].append(keyword.lower().strip())

    # Remove duplicates by converting to a set and back to a list
    for key, value in extracted_entities.items():
        extracted_entities[key] = list(set(value))

    return extracted_entities

# Extract relationships between entities based on proximity and assign weights
def extract_relationships(entities):
    relationships = defaultdict(int)

    # Generate relationships by creating pairs of different entities
    for entity_type, entity_list in entities.items():
        for ent1 in entity_list:
            for other_type, other_entity_list in entities.items():
                if entity_type != other_type:
                    for ent2 in other_entity_list:
                        relationships[(ent1, ent2)] += 1

    return relationships

# Build network graph using extracted entities and relationships
def build_graph(entities, relationships):
    G = nx.Graph()

    # Add entities as nodes with their types
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            G.add_node(entity, type=entity_type)

    # Add relationships as edges with weights
    for (source, target), weight in relationships.items():
        G.add_edge(source, target, weight=weight)

    return G

# Function to filter the graph to show only the direct neighbors of the selected country
def filter_graph_by_country(G, selected_country):
    sub_graph = nx.Graph()

    if selected_country in G.nodes:
        neighbors = list(G.neighbors(selected_country))
        sub_graph.add_node(selected_country, type=G.nodes[selected_country].get('type', ''))

        for neighbor in neighbors:
            sub_graph.add_node(neighbor, type=G.nodes[neighbor].get('type', ''))
            sub_graph.add_edge(selected_country, neighbor, weight=G[selected_country][neighbor]['weight'])

    return sub_graph

# Function to create and return the HTML data of PyVis graph
def visualize_interactive_graph(G, selected_country, degree_centrality, betweenness_centrality):
    net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white")

    # Define color mapping for entity types
    entity_colors = {
        'people': 'green',
        'countries': 'blue',
        'organizations': 'blue',
        'energy_types': 'yellow',
        'law': 'grey'
    }

    for node in G.nodes:
        node_type = G.nodes[node].get('type', 'organizations')

        if node == selected_country:
            color = 'red'
        else:
            color = entity_colors.get(node_type, "gray")

        size = max(10, degree_centrality[node] * 100)

        net.add_node(node, size=size, color=color,
                     title=f"{node}\nDegree: {degree_centrality[node]:.2f}\nBetweenness: {betweenness_centrality[node]:.2f}")

    for edge in G.edges():
        net.add_edge(edge[0], edge[1], title=f"Weight: {G[edge[0]][edge[1]]['weight']}",
                     value=G[edge[0]][edge[1]]['weight'])

    net.set_options("""{
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.01
        },
        "minVelocity": 0.75
      }
    }""")

    return net.generate_html()

# Function to read and combine text from multiple files
def read_and_combine_files(file_paths):
    combined_text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            combined_text += file.read() + "\n"
    return combined_text


# Sentiment Analysis for combined text
def analyze_combined_sentiment(file_paths):
    combined_text = read_and_combine_files(file_paths)
    blob = TextBlob(combined_text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


# Function to create and save word cloud as an image
def create_wordcloud_image(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plotting the word cloud
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjusted for better side-by-side layout
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    # Save the figure to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')  # Adjusted bbox_inches for better image cropping
    plt.close(fig)
    img.seek(0)
    return img

# Streamlit Dashboard
def main():
    st.set_page_config(layout="wide")
    st.title("Network Analysis Dashboard")

    # Tabs for different research questions
    tabs = st.tabs(["Research Question 1", "Research Question 2", "Research Question 3"])

    # Research Question 1 - Electricity Production
    with tabs[0]:
        st.header("How does public sentiment towards renewable energy vary across Australia, France, and Singapore, and what insights can be drawn regarding public support for renewable energy initiatives?")

        col1, col2 = st.columns(2)

        # Initialize combined sentiment data
        combined_sentiment_data = []

        # Analyze combined sentiment for each country and energy type
        for group, files in country_energy_files.items():
            country, energy_type = group.split(" - ")
            polarity, subjectivity = analyze_combined_sentiment(files)

            combined_sentiment_data.append({
                'Country': country,
                'Energy Type': energy_type,
                'Polarity': polarity,
                'Subjectivity': subjectivity
            })

        # Convert combined sentiment data into a DataFrame
        combined_sentiment_df = pd.DataFrame(combined_sentiment_data)

        fig_combined_polarity = px.bar(
            combined_sentiment_df,
            x="Country",
            y="Polarity",
            color="Energy Type",
            barmode="group",
            title="Sentiment Polarity by Country and Energy Type",
            labels={"Polarity": "Polarity"}
        )

        fig_combined_subjectivity = px.bar(
            combined_sentiment_df,
            x="Country",
            y="Subjectivity",
            color="Energy Type",
            barmode="group",
            title="Sentiment Subjectivity by Country and Energy Type",
            labels={"Subjectivity": "Subjectivity"}
        )

        with col1:
            st.plotly_chart(fig_combined_polarity)
            st.plotly_chart(choropleth_map, use_container_width=True)


        # Second column for the line chart
        with col2:
            st.plotly_chart(fig_combined_subjectivity)
            st.plotly_chart(line_chart, use_container_width=True)

    # Research Question 2 - Stakeholder Influence
    with tabs[2]:
        st.header(
            "Who are the central stakeholders and what are the primary energy types influencing the national energy policies of Australia, France, and Singapore?")

        st.markdown("""**Legend:**
        - <span style="color:red">**Selected Country**</span>
        - <span style="color:blue">**Organizations/Stakeholders**</span>
        - <span style="color:green">**People**</span>
        - <span style="color:yellow">**Energy Types**</span>
        - <span style="color:grey">**Policies**</span>
        """, unsafe_allow_html=True)

        selected_country = st.selectbox('Select a country to highlight:', ['All', 'Australia', 'Singapore', 'France'])

        if st.button('Highlight Influential Stakeholders'):
            if selected_country == "All":
                st.write("Showing all stakeholders")
            else:
                st.write(f"Showing direct stakeholders related to: {selected_country}")

            # Load documents
            document_paths = [
                os.path.join(dirname,"data/aus_stakeholder_info_1.txt"),
                os.path.join(dirname,"data/aus_stakeholder_info_2.txt"),
                os.path.join(dirname,"data/aus_stakeholder_info_3.txt"),
                os.path.join(dirname,"data/france_stakeholder_info_1.txt"),
                os.path.join(dirname,"data/singapore_stakeholder_info_1.txt"),
                os.path.join(dirname,"data/singapore_stakeholder_info_2.txt")
            ]

            documents = []
            for path in document_paths:
                with open(path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())

            all_entities = defaultdict(list)
            all_relationships = defaultdict(int)

            for doc_text in documents:
                preprocessed_text = preprocess_text(doc_text)
                entities = extract_entities(preprocessed_text)
                relationships = extract_relationships(entities)

                for entity_type, entity_list in entities.items():
                    all_entities[entity_type].extend(entity_list)
                for (source, target), weight in relationships.items():
                    all_relationships[(source, target)] += weight

            for entity_type, entity_list in all_entities.items():
                all_entities[entity_type] = list(set(entity_list))

            G = build_graph(all_entities, all_relationships)

            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)

            # Filter the graph based on the selected country
            G_filtered = filter_graph_by_country(G, selected_country.lower()) if selected_country != "All" else G

            # Visualize the filtered graph
            html_content = visualize_interactive_graph(G_filtered, selected_country.lower(), degree_centrality, betweenness_centrality)
            st.components.v1.html(html_content, height=1000)

    # Research Question 3 - Additional Analysis
    with tabs[1]:
        st.header(
            "How are the different energy sources progressed/regressed in the different regions and what can we predict about the key sources (Solar and Nuclear) in the coming years?")

        # Create two columns for side-by-side charts
        col1, col2 = st.columns(2)

        # First column for the electricity production line chart
        with col1:
            st.subheader("Past Production by Country and Source")
            country = st.selectbox("Select Country", selected_countries, index=0)
            sources = st.multiselect("Select Energy Sources", energy_sources, default=energy_sources)
            year_range = st.slider("Select Year Range", min_value=2008, max_value=2023, value=(2008, 2023))

            filtered_data_chart = long_format[
                (long_format['Entity'] == country) &
                (long_format['Source'].isin(sources)) &
                (long_format['Year'].between(year_range[0], year_range[1]))
                ]

            # Create line chart for production
            fig = go.Figure()

            for source in sources:
                df = filtered_data_chart[filtered_data_chart['Source'] == source]
                fig.add_trace(go.Scatter(x=df['Year'], y=df['Production'], mode='lines', name=source,
                                         line=dict(dash='dot' if source in ['Oil', 'Gas', 'Coal'] else 'solid')))

            # Update layout
            fig.update_layout(
                title=f'Electricity Production by Source in {country} ({year_range[0]}-{year_range[1]})',
                xaxis_title='Year',
                yaxis_title='Electricity Production (TWh)',
                legend_title='Energy Source',
                plot_bgcolor='white',
                xaxis=dict(showline=True, showgrid=True, linecolor='black'),
                yaxis=dict(showline=True, showgrid=True, gridcolor='lightgrey', linecolor='black')
            )

            st.plotly_chart(fig)

        # Second column for the forecast production chart
        with col2:
            st.subheader("Forecast Production by Country and Source")

            forecast_country = st.selectbox("Select Country for Forecast", selected_countries)
            forecast_source = st.selectbox("Select Energy Source for Forecast", energy_sources)

            # Filter data for the selected country and source
            forecast_data = filtered_data[(filtered_data['Entity'] == forecast_country) &
                                          (filtered_data['Source'] == forecast_source)]

            # Create a figure for historical and forecasted production
            forecast_fig = go.Figure()

            # Add historical data line
            forecast_fig.add_trace(go.Scatter(
                x=forecast_data['Year'],
                y=forecast_data['Production'],
                mode='lines',
                name=f'{forecast_country} - {forecast_source} (Historical)'
            ))

            # Add forecasted data line (if it exists)
            if 'Forecast' in forecast_data.columns:
                forecast_fig.add_trace(go.Scatter(
                    x=forecast_data['Year'],
                    y=forecast_data['Forecast'],
                    mode='lines',
                    name=f'{forecast_country} - {forecast_source} (Forecast)',
                    line=dict(dash='dash')
                ))

            # Update layout for the forecast chart
            forecast_fig.update_layout(
                title=f'Production and Forecast for {forecast_country} - {forecast_source}',
                xaxis_title='Year',
                yaxis_title='Electricity Production (TWh)',
                plot_bgcolor='white',
                xaxis=dict(showline=True, showgrid=True, linecolor='black'),
                yaxis=dict(showline=True, showgrid=True, gridcolor='lightgrey', linecolor='black')
            )

            st.plotly_chart(forecast_fig)

    # with tabs[3]:
    #     st.subheader("Policy Documents Common Terms")
    #     # Create three columns for side-by-side layout
    #     col4, col5, col6 = st.columns(3)
    #
    #     # Read and combine texts for each country and generate word clouds
    #     countries = {
    #         "Australia": (Australia_Policy, col4),
    #         "France": (France_Policy, col5),
    #         "Singapore": (Singapore_Policy, col6)
    #     }
    #
    #     for country, (files, column) in countries.items():
    #         combined_text = read_and_combine_files(files)
    #         wordcloud_image = create_wordcloud_image(combined_text)
    #
    #         # Display the word cloud image in the respective column
    #         with column:
    #             st.image(wordcloud_image, caption=f'{country} Policy Word Cloud', use_column_width=True)


if __name__ == "__main__":
    main()
