import pandas as pd
import datetime
import numpy as np
import networkx as nx
from ipysigma import Sigma

# imports
# partnerships
long_df = pd.read_csv('../Data/innovation_network_prepost_2020-01-01.csv')
# convert the column post_date to datetime
long_df['post_date'] = pd.to_datetime(long_df['post_date'])

# organizations
df2 = pd.read_csv("../Data/organizations_prepost_2020-01-01.csv")

####Drop Duplicates of interactions:
columns = long_df.columns[6:]

columns_= list(columns)
columns_.append('partners')

long_df["partners"] = long_df.apply(lambda x: '-'.join(sorted([x['source'], x['target']])), axis=1)

"""Split the datasets into pre IRA and post IRA"""
# df with datatime before 1.07.2022
df_pre_ira = long_df[long_df['post_date'] < datetime.datetime(2022, 7, 1)]
# drop duplicate columns
df_pre_ira = df_pre_ira.drop_duplicates(subset=columns_)
df_pre_ira = df_pre_ira.drop(columns=['partners'])

df_pos_ira = long_df[long_df['post_date'] >= datetime.datetime(2022, 7, 1)]
df_pos_ira = df_pos_ira.drop_duplicates(subset=columns_)
df_pos_ira = df_pos_ira.drop(columns=['partners'])

# list of all technologies to be plotted
list_techs= ["All", "Biomass", "Biofuels", "Biogas", "Wind", "Offshore_Wind",
                   "Solar", "Concentrated_Solar", "Waste_to_Heat", "Direct_Air_Capture", "Carbon_Capture_And_Storage", "Biochar", "BECCS",
                   "Carbon_Direct_Removal", "Hydrogen", "Nuclear_Energy", "Nuclear_Fusion", "Hydro_Energy",
                   "Geothermal", "Battery", "Electric_Vehicles", "Sustainable_Aviation_Fuels", "E_Fuels",
                   "Marine_Energy", "Heat_Pumps", "Railway", "Electric_Shipping", "Electric_Aviation","Fuel_Cell_Aviation"]

# list of all orgtypes to be colored separately
list_orgtypes = ["All", "Green services", "Service sector", "Mining industry", "Green industry", "Utilities", "Oil and gas firms", "Other industry", "Research organizations", "Banks", "Venture Capital", "Other Finance", "Governmental organizations", "Incubators/Accelerators",  "Other"]


for period, df in enumerate([df_pre_ira,df_pos_ira]):
    list_number_nodes = []
    list_number_edges = []
    tech_rows = []
    org_rows = []

    for tech in list_techs:
        if tech=="All":
            df_copy = df.copy()

        else:
            df_copy = df[df[tech]==1]


        G = nx.from_pandas_edgelist(df_copy, source="source", target="target")
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
            './network_period_'+str(period)+'_'+ tech+'.html',
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


