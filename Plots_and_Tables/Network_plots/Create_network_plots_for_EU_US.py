import pandas as pd
import datetime
from itertools import combinations
import ast
import re
import numpy as np
import networkx as nx
from ipysigma import Sigma

# as "domestic" link counts:
# both organizations need a location in the country/continent
# at least one needs a headquarter location in the country/continent

# technologies to analyze:
list_techs= ["All", "Biomass", "Biofuels", "Biogas", "Wind", "Offshore_Wind",
                   "Solar", "Concentrated_Solar", "Waste_to_Heat", "Direct_Air_Capture", "Carbon_Capture_And_Storage", "Biochar", "BECCS",
                   "Carbon_Direct_Removal", "Hydrogen", "Nuclear_Energy", "Nuclear_Fusion", "Hydro_Energy",
                   "Geothermal", "Battery", "Electric_Vehicles", "Sustainable_Aviation_Fuels", "E_Fuels",
                   "Marine_Energy", "Heat_Pumps", "Railway", "Electric_Shipping", "Electric_Aviation","Fuel_Cell_Aviation"]

# list of orgtypes
list_orgtypes = ["All", "Green services", "Service sector", "Mining industry", "Green industry", "Utilities", "Oil and gas firms", "Other industry", "Research organizations", "Banks", "Venture Capital", "Other Finance", "Governmental organizations", "Incubators/Accelerators",  "Other"]


#to start: get all locations
countries_eu = [
    'ES', 'BE', 'AT', 'NL', 'FR', 'IE', 'IT', 'FI', 'DE', 'SE', 'GR',
    'PT', 'LU', 'PL', 'DK', 'SK', 'CZ', 'BG', 'HR', 'HU', 'EE', 'LT',
    'LV', 'CY', 'SI', 'RO', 'MT'
]

list_US = ["US"]

# import all locations of organizations (including non headquarter locations)
df_locations = pd.read_csv("../Data/locations.csv")

# get all Linkedin_names
organizations_location_EU = list(df_locations["Linkedin_name"][df_locations["country"].isin(countries_eu)].drop_duplicates())
organizations_location_US = list(df_locations["Linkedin_name"][df_locations["country"].isin(list_US)].drop_duplicates())

# get all Linkedin_names with headquarter
organizations_headquarter_EU = list(df_locations["Linkedin_name"][(df_locations["country"].isin(countries_eu)) & (df_locations["headquarter"] == 1)].drop_duplicates())
organizations_headquarter_US = list(df_locations["Linkedin_name"][(df_locations["country"].isin(list_US)) & (df_locations["headquarter"] == 1)].drop_duplicates())

# get the dataframe with the links
long_df = pd.read_csv("../Data/innovation_network_prepost_2020-01-01.csv")

# get the dataframe with the organizations
df2 = pd.read_csv("../Data/organizations_prepost_2020-01-01.csv")

# only include the period after the IRA
#make post_date a datetime format
long_df['post_date'] = pd.to_datetime(long_df['post_date'])
long_df = long_df[long_df['post_date'] >= datetime.datetime(2022, 7, 1)]

####Drop Duplicates of interactions:
columns = long_df.columns[6:]

columns_= list(columns)
columns_.append('partners')

long_df["partners"] = long_df.apply(lambda x: '-'.join(sorted([x['source'], x['target']])), axis=1)

# drop duplicates
long_df = long_df.drop_duplicates(subset=columns_)
long_df = long_df.drop(columns=["partners"])

list_number_nodes = []
list_number_edges = []
tech_rows = []
org_rows = []
list_collabtypes = []
# let's start with the EU:
for technology in list_techs:
    if technology=="All":
        df_tech = long_df.copy()
    else:
        df_tech = long_df[long_df[technology] == 1]

    # for each row in long_df, check if the source and target are in the EU
    df_tech = df_tech[(df_tech.source.isin(organizations_location_EU)) & (df_tech.target.isin(organizations_location_EU))]

    # additionally, check if at least one of the two has a headquarter in the EU
    df_tech = df_tech[(df_tech.source.isin(organizations_headquarter_EU)) | (df_tech.target.isin(organizations_headquarter_EU))]

    G = nx.from_pandas_edgelist(df_tech, source="source", target="target")
    # Compute the node degree for node size
    node_degrees = dict(G.degree())

    # Compute degree centrality
    degree_centrality = nx.degree_centrality(G)

    # get the max degree centrality
    try:
        max_degree_centrality = max(degree_centrality.values())
    except:
        max_degree_centrality = np.nan

    # Map firm types from df2 to the nodes
    firm_type_dict = pd.Series(df2['orgtype'].values, index=df2['Linkedin_name']).to_dict()
    label_dict = pd.Series(df2['name'].values, index=df2['Linkedin_name']).to_dict()

    # Set node degree and firm type as node attributes
    nx.set_node_attributes(G, node_degrees, 'degree')
    nx.set_node_attributes(G, firm_type_dict, 'firm_type')
    nx.set_node_attributes(G, label_dict, 'label')

    # Define a color mapping for firm types
    color_mapping = {
        'Service sector': 'lightblue',
        'Green services': 'lightgreen',
        'Mining industry': '#756bb1',
        'Green industry': 'green',
        'Research organizations': 'orange',
        'Utilities': '#104862',
        'Oil and gas firms': 'brown',
        'Other Finance': 'pink',
        'Banks': 'purple',
        'Venture Capital': 'yellow',
        'Governmental organizations': 'red',
        'Incubators/Accelerators': '#008080',
        'Other industry': 'blue'
    }

    # Group nodes by orgtype
    orgtype_groups = {}
    for node, data in G.nodes(data=True):
        orgtype = data.get('firm_type', 'Other')
        orgtype_groups.setdefault(orgtype, []).append(node)

    #order orgtype groups by a list if list item is not in orgtype groups, ignore it
    orgtype_groups = {k: orgtype_groups[k] for k in list_orgtypes[1:] if k in orgtype_groups}


    # Calculate angular allocation for each orgtype
    total_nodes = len(G.nodes)
    orgtype_angles = {orgtype: len(nodes) / total_nodes * 360 for orgtype, nodes in orgtype_groups.items()}


    # Generate positions
    def orgtype_theta_layout(G, centrality, orgtype_angles, orgtype_groups):
        layout = {}
        start_angle = 0  # Starting angle for the first orgtype

        for orgtype, nodes in orgtype_groups.items():
            # Angular space for this orgtype
            angle_span = orgtype_angles[orgtype]
            end_angle = start_angle + angle_span

            # Generate random angles within the allocated space
            angles = np.linspace(start_angle, end_angle, len(nodes), endpoint=False)
            np.random.shuffle(angles)  # Randomize node positions within the space

            for node, theta in zip(nodes, angles):
                # Scale radius based on centrality (higher centrality = smaller radius)
                centrality_value = centrality[node]
                # normalize the centrality value by the max degree centrality, so the highest value is 1
                centrality_value = (centrality_value / max_degree_centrality) - 0.1
                radius = (1 - centrality_value) * 10  # Adjust max radius as needed

                # Convert polar to Cartesian coordinates
                x = radius * np.cos(np.radians(theta))
                y = radius * np.sin(np.radians(theta))
                layout[node] = (x, y)

            start_angle = end_angle  # Update start_angle for the next orgtype

        return layout


    # Generate layout
    theta_layout = orgtype_theta_layout(G, degree_centrality, orgtype_angles, orgtype_groups)

    # Assign positions to nodes
    for node, (x, y) in theta_layout.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y

    # Visualize the graph using Sigma
    Sigma.write_html(
        G,
        'EU_network_'+ technology+'.html',
        fullscreen=True,
        node_metrics=['louvain'],
        node_color='firm_type',
        node_size_range=(0.1, 50),
        max_categorical_colors=10,
        default_edge_type='curve',
        node_border_color_from='node',
        default_node_label_size=14,  # Node label font size
        node_size='degree',
        node_color_palette=color_mapping,
        default_edge_size=0.08,
        node_label="label",
        label_density=0,  # Use the 'label' attribute for node labels
        layout_settings={
            "defaultNodePosition": ["x", "y"],  # Use the precomputed x, y layout
        }
    )


    # number of nodes
    list_number_nodes.append(len(G.nodes))
    # number of edges
    list_number_edges.append(len(G.edges))
    # tech
    tech_rows.append(technology)

    # add the distribution over the collaboration types
    # take the last 23 columns of df_tech
    column_names = df_tech.columns[-23:]
    df_tech_collabtypes = df_tech[column_names ].sum()
    list_collabtypes.append(df_tech_collabtypes)


# create dataframe with the descriptives
df_descriptives_EU = pd.DataFrame({"tech":tech_rows, "number_nodes":list_number_nodes, "number_edges":list_number_edges,  })

# create a dataframe df_collabtypes that has the column_names as columns and the values of the collabtypes as rows
df_collabtypes = pd.DataFrame(list_collabtypes, columns=column_names)

# add the collabtypes to the descriptives
df_descriptives_EU = pd.concat([df_descriptives_EU, df_collabtypes], axis=1)

# save as csv for period
df_descriptives_EU.to_csv("descriptives_EU.csv", index=False)


##### now for the US
print("US")
list_number_nodes = []
list_number_edges = []
tech_rows = []
list_collabtypes = []
# let's start with the US:
for technology in list_techs:
    if technology=="All":
        df_tech = long_df.copy()
    else:
        df_tech = long_df[long_df[technology] == 1]

    # for each row in long_df, check if the source and target are in the US
    df_tech = df_tech[(df_tech.source.isin(organizations_location_US)) & (df_tech.target.isin(organizations_location_US))]

    # additionally, check if at least one of the two has a headquarter in the US
    df_tech = df_tech[(df_tech.source.isin(organizations_headquarter_US)) | (df_tech.target.isin(organizations_headquarter_US))]

    G = nx.from_pandas_edgelist(df_tech, source="source", target="target")
    # Compute the node degree for node size
    node_degrees = dict(G.degree())

    # Compute degree centrality
    degree_centrality = nx.degree_centrality(G)

    # get the max degree centrality
    try:
        max_degree_centrality = max(degree_centrality.values())
    except:
        max_degree_centrality = np.nan

    # Map firm types from df2 to the nodes
    firm_type_dict = pd.Series(df2['orgtype'].values, index=df2['Linkedin_name']).to_dict()
    label_dict = pd.Series(df2['name'].values, index=df2['Linkedin_name']).to_dict()

    # Set node degree and firm type as node attributes
    nx.set_node_attributes(G, node_degrees, 'degree')
    nx.set_node_attributes(G, firm_type_dict, 'firm_type')
    nx.set_node_attributes(G, label_dict, 'label')

    # Define a color mapping for firm types
    color_mapping = {
        'Service sector': 'lightblue',
        'Green services': 'lightgreen',
        'Mining industry': '#756bb1',
        'Green industry': 'green',
        'Research organizations': 'orange',
        'Utilities': '#104862',
        'Oil and gas firms': 'brown',
        'Other Finance': 'pink',
        'Banks': 'purple',
        'Venture Capital': 'yellow',
        'Governmental organizations': 'red',
        'Incubators/Accelerators': '#008080',
        'Other industry': 'blue'
    }

    # Group nodes by orgtype
    orgtype_groups = {}
    for node, data in G.nodes(data=True):
        orgtype = data.get('firm_type', 'Other')
        orgtype_groups.setdefault(orgtype, []).append(node)

    #order orgtype groups by a list if list item is not in orgtype groups, ignore it
    orgtype_groups = {k: orgtype_groups[k] for k in list_orgtypes[1:] if k in orgtype_groups}


    # Calculate angular allocation for each orgtype
    total_nodes = len(G.nodes)
    orgtype_angles = {orgtype: len(nodes) / total_nodes * 360 for orgtype, nodes in orgtype_groups.items()}


    # Generate positions
    def orgtype_theta_layout(G, centrality, orgtype_angles, orgtype_groups):
        layout = {}
        start_angle = 0  # Starting angle for the first orgtype

        for orgtype, nodes in orgtype_groups.items():
            # Angular space for this orgtype
            angle_span = orgtype_angles[orgtype]
            end_angle = start_angle + angle_span

            # Generate random angles within the allocated space
            angles = np.linspace(start_angle, end_angle, len(nodes), endpoint=False)
            np.random.shuffle(angles)  # Randomize node positions within the space

            for node, theta in zip(nodes, angles):
                # Scale radius based on centrality (higher centrality = smaller radius)
                centrality_value = centrality[node]
                # normalize the centrality value by the max degree centrality, so the highest value is 1
                centrality_value = (centrality_value / max_degree_centrality) - 0.1
                radius = (1 - centrality_value) * 10  # Adjust max radius as needed

                # Convert polar to Cartesian coordinates
                x = radius * np.cos(np.radians(theta))
                y = radius * np.sin(np.radians(theta))
                layout[node] = (x, y)

            start_angle = end_angle  # Update start_angle for the next orgtype

        return layout


    # Generate layout
    theta_layout = orgtype_theta_layout(G, degree_centrality, orgtype_angles, orgtype_groups)

    # Assign positions to nodes
    for node, (x, y) in theta_layout.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y

    # Visualize the graph using Sigma
    Sigma.write_html(
        G,
        'US_network_'+ technology+'.html',
        fullscreen=True,
        node_metrics=['louvain'],
        node_color='firm_type',
        node_size_range=(0.1, 50),
        max_categorical_colors=10,
        default_edge_type='curve',
        node_border_color_from='node',
        default_node_label_size=14,  # Node label font size
        node_size='degree',
        node_color_palette=color_mapping,
        default_edge_size=0.08,
        node_label="label",
        label_density=0,  # Use the 'label' attribute for node labels
        layout_settings={
            "defaultNodePosition": ["x", "y"],  # Use the precomputed x, y layout
        }
    )


    # number of nodes
    list_number_nodes.append(len(G.nodes))
    # number of edges
    list_number_edges.append(len(G.edges))
    # tech
    tech_rows.append(technology)

    # add the distribution over the collaboration types
    # take the last 23 columns of df_tech
    column_names = df_tech.columns[-23:]
    df_tech_collabtypes = df_tech[column_names ].sum()
    list_collabtypes.append(df_tech_collabtypes)


# create dataframe with the descriptives
df_descriptives_US = pd.DataFrame({"tech":tech_rows, "number_nodes":list_number_nodes, "number_edges":list_number_edges,  })

# create a dataframe df_collabtypes that has the column_names as columns and the values of the collabtypes as rows
df_collabtypes = pd.DataFrame(list_collabtypes, columns=column_names)

# add the collabtypes to the descriptives
df_descriptives_US = pd.concat([df_descriptives_US, df_collabtypes], axis=1)

# save as csv for period
df_descriptives_US.to_csv("descriptives_US.csv", index=False)
