import spacy
import re
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
import streamlit as st

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

# Step 2: Entity Extraction combining SpaCy and keyword-based extraction
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

# Step 3: Extract relationships between entities based on proximity and assign weights
def extract_relationships(entities):
    relationships = defaultdict(int)  # Using defaultdict to count co-occurrences

    # Generate relationships by creating pairs of different entities
    for entity_type, entity_list in entities.items():
        for ent1 in entity_list:
            for other_type, other_entity_list in entities.items():
                if entity_type != other_type:
                    for ent2 in other_entity_list:
                        relationships[(ent1, ent2)] += 1  # Increment count for each co-occurrence

    return relationships

# Step 4: Build network graph using extracted entities and relationships
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
        neighbors = list(G.neighbors(selected_country))  # Get all nodes connected to the selected country
        sub_graph.add_node(selected_country, type=G.nodes[selected_country].get('type', ''))

        # Add neighbors and their connections to the subgraph
        for neighbor in neighbors:
            sub_graph.add_node(neighbor, type=G.nodes[neighbor].get('type', ''))
            sub_graph.add_edge(selected_country, neighbor, weight=G[selected_country][neighbor]['weight'])

    return sub_graph

# Function to create and return the HTML content of PyVis graph
def visualize_interactive_graph(G, selected_country, degree_centrality, betweenness_centrality):
    net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white")

    # Define color mapping for entity types
    entity_colors = {
        'people': 'green',
        'countries': 'blue',  # Non-selected countries
        'organizations': 'blue',  # Organizations (including stakeholders and government bodies)
        'energy_types': 'yellow',
        'law': 'grey'  # Policies
    }

    for node in G.nodes:
        node_type = G.nodes[node].get('type', 'organizations')

        # Make the selected country red
        if node == selected_country:
            color = 'red'
        else:
            color = entity_colors.get(node_type, "gray")

        # Size based on degree centrality from full graph, not the filtered graph
        size = max(10, degree_centrality[node] * 100)

        # Add node with degree centrality and betweenness as title
        net.add_node(node, size=size, color=color,
                     title=f"{node}\nDegree: {degree_centrality[node]:.2f}\nBetweenness: {betweenness_centrality[node]:.2f}")

    for edge in G.edges():
        # Add edge with weight as title
        net.add_edge(edge[0], edge[1], title=f"Weight: {G[edge[0]][edge[1]]['weight']}",
                     value=G[edge[0]][edge[1]]['weight'])

    net.set_options("""
    {
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
    }
    """)

    return net.generate_html()

# Streamlit Dashboard
def main():
    st.title("Stakeholder Influence on Energy Policies")

    # Display legend on the dashboard
    st.markdown("""
    **Legend:**
    - <span style="color:red">**Selected Country**</span>
    - <span style="color:blue">**Organizations/Stakeholders**</span>
    - <span style="color:green">**People**</span>
    - <span style="color:yellow">**Energy Types**</span>
    - <span style="color:grey">**Policies**</span>
    """, unsafe_allow_html=True)

    # Limit to only specific countries: Australia, Singapore, France
    selected_country = st.selectbox('Select a country to highlight:', ['All', 'Australia', 'Singapore', 'France'])

    # Button to highlight influential stakeholders
    if st.button('Highlight Influential Stakeholders'):
        if selected_country == "All":
            st.write("Showing all stakeholders")
        else:
            st.write(f"Showing direct stakeholders related to: {selected_country}")

        # Load documents
        document_paths = [
            "aus_stakeholder_info_1.txt",
            "aus_stakeholder_info_2.txt",
            "aus_stakeholder_info_3.txt",
            'france_stakeholder_info_1.txt',
            'singapore_stakeholder_info_1.txt',
            'singapore_stakeholder_info_2.txt'
        ]

        # Load documents
        documents = []
        for path in document_paths:
            with open(path, 'r', encoding='utf-8') as file:
                documents.append(file.read())

        all_entities = defaultdict(list)
        all_relationships = defaultdict(int)

        # Process each document
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

        # Build the complete graph
        G = build_graph(all_entities, all_relationships)

        # Calculate degree centrality and betweenness centrality on the full graph
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        # Filter the graph based on the selected country
        if selected_country != "All":
            G_filtered = filter_graph_by_country(G, selected_country.lower())  # Make sure it's lowercase for matching
        else:
            G_filtered = G

        # Visualize the filtered graph and get HTML content
        html_content = visualize_interactive_graph(G_filtered, selected_country.lower(), degree_centrality, betweenness_centrality)

        # Display the interactive graph in Streamlit
        st.components.v1.html(html_content, height=1000)

if __name__ == "__main__":
    main()
