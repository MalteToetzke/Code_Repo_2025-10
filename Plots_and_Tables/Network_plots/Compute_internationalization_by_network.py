import pandas as pd
import datetime
import numpy as np

"""This code analyzes the share of domestic partnerships (i.e., both organizations have locations in the same country) 
based on 
1. headquarters and 
2. any possible location

Depending on the hardware used, the code runs for > 12 hours
"""
# partnerships
long_df = pd.read_csv('../Data/innovation_network_prepost_2020-01-01.csv')

# organizations
df2 = pd.read_csv("../Data/organizations_prepost_2020-01-01.csv")

# Create dictionaries for headquarter country and organization type based on Linkedin names
name_country_dict = pd.Series(df2['country'].values, index=df2['Linkedin_name']).to_dict()

name_orgtype_dict = pd.Series(df2['orgtype'].values, index=df2['Linkedin_name']).to_dict()


# import the countries of all organization locations (not only headquarters)
country_codes = pd.read_csv('../Data/country_codes.csv')
print(country_codes.head())


#convert post_date to datetime
long_df['post_date'] = pd.to_datetime(long_df['post_date'])

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

# add columns for each technology
list_techs= ["All", "Biomass", "Biofuels", "Biogas", "Wind", "Offshore_Wind",
                   "Solar", "Concentrated_Solar", "Waste_to_Heat", "Direct_Air_Capture", "Carbon_Capture_And_Storage", "Biochar", "BECCS",
                   "Carbon_Direct_Removal", "Hydrogen", "Nuclear_Energy", "Nuclear_Fusion", "Hydro_Energy",
                   "Geothermal", "Battery", "Electric_Vehicles", "Sustainable_Aviation_Fuels", "E_Fuels",
                   "Marine_Energy", "Heat_Pumps", "Railway", "Electric_Shipping", "Electric_Aviation","Fuel_Cell_Aviation"]

list_orgtypes = ["All", "Service sector", "Green services", "Mining industry", "Green industry", "Research organizations", "Utilities", "Oil and gas firms", "Other Finance", "Banks", "Venture Capital", "Governmental organizations", "Incubators/Accelerators", "Other industry"]


# for each row in long_df, check if the source and target are in the country_codes df and have a row where the country is the same
df_pre_ira["same_country_any"] = False
df_pre_ira["drop_row_country_any"] = False

country_codes_pre = country_codes[country_codes["updateCount"] == 1]
for index, row in df_pre_ira.iterrows():
    if index% 1000 == 0:
        print(index)
    source_country = list(country_codes_pre[country_codes_pre["Linkedin_name"] == row["source"]]["country"])
    target_country = list(country_codes_pre[country_codes_pre["Linkedin_name"] == row["target"]]["country"])
    #if any of the two lists is empty, skip
    if len(source_country) == 0 or len(target_country) == 0:

        df_pre_ira.at[index, "drop_row_country_any"] = True
    #check if any countries in the two lists are the same
    if len(set(source_country).intersection(target_country)) > 0:
        df_pre_ira.at[index, "same_country_any"] = True

df_pre_ira["source_hq"] = df_pre_ira["source"].map(name_country_dict)
df_pre_ira["target_hq"] = df_pre_ira["target"].map(name_country_dict)
df_pre_ira["same_country_hq"] = False
for index, row in df_pre_ira.iterrows():
    if row["source_hq"] == row["target_hq"]:
        df_pre_ira.at[index, "same_country_hq"] = True

df_pre_ira["orgtype_source"] = df_pre_ira["source"].map(name_orgtype_dict)
df_pre_ira["orgtype_target"] = df_pre_ira["target"].map(name_orgtype_dict)


list_perc_national_hq = []
list_perc_national_any = []
tech_rows = []
org_rows = []

for tech in list_techs:
    if tech=="All":
        df_copy = df_pre_ira.copy()

    elif tech=="Bioenergy":
        df_copy = df_pre_ira[df_pre_ira[["Biomass", "Biofuels", "Biogas"]].sum(axis=1)>0]

    else:
        df_copy = df_pre_ira[df_pre_ira[tech]==1]

    for orgtype in list_orgtypes:
        if orgtype == "All":
            # add percentage of national edges of hqs
            try:
                list_perc_national_hq.append(df_copy[(df_copy['same_country_hq'] == True)].shape[0] / df_copy.shape[0])
            except ZeroDivisionError:
                list_perc_national_hq.append(np.nan)
            # add percentage of national edges of any
            try:
                list_perc_national_any.append(df_copy[(df_copy['same_country_any'] == True)].shape[0] / df_copy.shape[0])
            except ZeroDivisionError:
                list_perc_national_any.append(np.nan)

            tech_rows.append(tech)
            org_rows.append(orgtype)
        else:
            # add percentage of national edges of hqs
            try:
                list_perc_national_hq.append(df_copy[(df_copy['same_country_hq'] == True) & ((df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype))].shape[0] / df_copy[(df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype)].shape[0])
            except ZeroDivisionError:
                list_perc_national_hq.append(np.nan)
            # add percentage of national edges of any
            try:
                list_perc_national_any.append(df_copy[(df_copy['same_country_any'] == True) & ((df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype))].shape[0] / df_copy[(df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype)].shape[0])
            except ZeroDivisionError:
                list_perc_national_any.append(np.nan)

            tech_rows.append(tech)
            org_rows.append(orgtype)

df_perc_pre = pd.DataFrame({"tech": tech_rows, "orgtype": org_rows, "perc_national_hq": list_perc_national_hq, "perc_national_any": list_perc_national_any})
df_perc_pre.to_csv('percentage_national_edges_pre.csv')


# for each row in long_df, check if the source and target are in the country_codes df and have a row where the country is the same
df_pos_ira["same_country_any"] = False
df_pos_ira["drop_row_country_any"] = False

country_codes_pos = country_codes[country_codes["updateCount"] == 1]
for index, row in df_pos_ira.iterrows():
    if index % 1000 == 0:
        print(index)
    source_country = list(country_codes_pos[country_codes_pos["Linkedin_name"] == row["source"]]["country"])
    target_country = list(country_codes_pos[country_codes_pos["Linkedin_name"] == row["target"]]["country"])
    # If any of the two lists is empty, mark row for dropping
    if len(source_country) == 0 or len(target_country) == 0:
        df_pos_ira.at[index, "drop_row_country_any"] = True
    # Check if any countries in the two lists overlap
    if len(set(source_country).intersection(target_country)) > 0:
        df_pos_ira.at[index, "same_country_any"] = True

df_pos_ira["source_hq"] = df_pos_ira["source"].map(name_country_dict)
df_pos_ira["target_hq"] = df_pos_ira["target"].map(name_country_dict)
df_pos_ira["same_country_hq"] = False
for index, row in df_pos_ira.iterrows():
    if row["source_hq"] == row["target_hq"]:
        df_pos_ira.at[index, "same_country_hq"] = True

df_pos_ira["orgtype_source"] = df_pos_ira["source"].map(name_orgtype_dict)
df_pos_ira["orgtype_target"] = df_pos_ira["target"].map(name_orgtype_dict)


list_perc_national_hq_pos = []
list_perc_national_any_pos = []
tech_rows_pos = []
org_rows_pos = []

for tech in list_techs:
    if tech == "All":
        df_copy = df_pos_ira.copy()

    else:
        df_copy = df_pos_ira[df_pos_ira[tech] == 1]

    for orgtype in list_orgtypes:
        if orgtype == "All":
            # Add percentage of national edges (hq)
            try:
                list_perc_national_hq_pos.append(df_copy[(df_copy['same_country_hq'] == True)].shape[0] / df_copy.shape[0])
            except ZeroDivisionError:
                list_perc_national_hq_pos.append(np.nan)
            # Add percentage of national edges (any)
            try:
                list_perc_national_any_pos.append(df_copy[(df_copy['same_country_any'] == True)].shape[0] / df_copy.shape[0])
            except ZeroDivisionError:
                list_perc_national_any_pos.append(np.nan)

            tech_rows_pos.append(tech)
            org_rows_pos.append(orgtype)
        else:
            # Percentage national edges (hq) for given orgtype
            try:
                list_perc_national_hq_pos.append(
                    df_copy[(df_copy['same_country_hq'] == True) & ((df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype))].shape[0] /
                    df_copy[(df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype)].shape[0]
                )
            except ZeroDivisionError:
                list_perc_national_hq_pos.append(np.nan)
            # Percentage national edges (any) for given orgtype
            try:
                list_perc_national_any_pos.append(
                    df_copy[(df_copy['same_country_any'] == True) & ((df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype))].shape[0] /
                    df_copy[(df_copy['orgtype_source'] == orgtype) | (df_copy['orgtype_target'] == orgtype)].shape[0]
                )
            except ZeroDivisionError:
                list_perc_national_any_pos.append(np.nan)

            tech_rows_pos.append(tech)
            org_rows_pos.append(orgtype)

df_perc_pos = pd.DataFrame({
    "tech": tech_rows_pos,
    "orgtype": org_rows_pos,
    "perc_national_hq": list_perc_national_hq_pos,
    "perc_national_any": list_perc_national_any_pos
})

df_perc_pos.to_csv('percentage_national_edges_pos.csv', index=False)




