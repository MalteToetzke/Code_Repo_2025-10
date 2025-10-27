# ClimateTech_Innovation_Networks_Code
This is the code used for the information extraction and analyses in the paper on climate-tech innovation networks

## SETUP
To run the code, you need to have Python 3.10 or higher installed. The code has been tested on Python 3.13

To install the required packages, run the following command in your terminal:
```
pip install requirements.txt
```

## ITEMS

### Classifiers
Replicate the training and validation of the classifiers used in the paper. Note: the relevance classifier (relevane_classifier_climatetech.py) automatically downloads the underlying transformer model at runtime from https://huggingface.co/climatebert/distilroberta-base-climate-f, if it is not already stored locally. For all other classification scripts, the DeepSeek model requires to setup a token for accessing the API. 

Run the following commands from the Shell to run the classifiers:
``` 
python Classifiers/relevance_classifier_climatech.py  # trains the relevance classifier on manually annotated train and validation data and evaluates it on the test set
python Classifiers/Collaboration_classifier.py  # runs the collaboration classifier to classify LinkedIn posts and compare with manually annotated data
python Classifiers/organization_type_classifier.py  # runs the organization type classifier to classify LinkedIn company profiles and compare with manually annotated data
python Classifiers/sector.py  # runs the sector classifier to classify LinkedIn company profiles and compare with manually annotated data
```
All outputs are stored in the base folder.

#### Data
Data used for validation and training of classifications scripts




### Plots_and_Tables
Scripts to replicate the plots and tables of the paper. Data imported in the scripts is stored in the Data folder

#### Data
The data used to replicate plots, tables, and descriptives stated in the paper (run through the scripts below)

Note: the files were too large to be included in Github. Download them from the link below and place them in the Data folder:

Download link: https://polybox.ethz.ch/index.php/s/yHoCqw5zGAxfTYm (Password is stated in the Data Availability statement of the Manuscript)

Important: The data must only be used exclusively for validation purposes! Sharing the data or using the data for other purposes is strictly prohibited!



##### Overview of Data
- country_codes.csv: Dataframe linking organizations to country codes of locations
- gdp.csv: Dataframe linking countries to GDP data
- innovation_networks_prepost_2020-01-01.csv: All climate-tech innovation partnerships between 2020 and 2024. 
- linkedin-users-by-country-2024.csv: Dataframe linking countries to the number of LinkedIn users in 2024
- LinkedIn_Orgs_with_Websites: Dataframe linking LinkedIn organizations to their websites and other metadata used to link with other databases (e.g., i3, Crunchbase)
- locations.csv: Dataframe linking organizations to their locations
- locations_with_postalcodes.csv: Dataframe linking organizations to their locations with postalcodes -- used to map onto US states
- ne_110m_admin_0_countries.csv: Dataframe with country names and codes used to map geodata
- organizations_prepost_2020-01-01.csv: All climate-tech organizations involved in innovation partnerships between 2020 and 2024
- population.csv: Dataframe linking countries to population data
- tech_country_growth.csv: Dataframe with descriptives on the growth of innovation networks over the Pre-IRA and Post-IRA periods, for each technology and country


#### Governmental_involvement
Scripts to replicate the analysis of governmental involvement in climate-tech innovation networks, specifically for Section 3 of the paper
- compute_governmental_involvement.py: Script to compute the governmental involvement in climate-tech innovation networks (Figure 3)
- regression_government_domestic_growth.py: Script to run the regression analysis conducted in the Supplementary Discussion S4

#### Network_plots
Scripts to replicate the network plots of the paper, specifically for Section 2,4,5 of the paper

- Berlin_Houston_Hydrogen.ipynb: Script to replicate the Berlin-Houston hydrogen innovation network plot (Figure 5)
- Compute_internationalization_by_network.py: Script to compute the share of domestic vs. international partnerships, referenced in Section 2 of the paper
- Create_network_plots.py: Script to create the network plots of the paper (Figure 2, Supplementary Figures S8--S35)
- Create_network_plots_for_EU_US.py: Script to create the network plots of the paper for the EU and US (Figure 4)

#### US_EU_Connectivity_and_Specialization
Scripts to replicate the analysis between States/Country connectivity in the US/EU and specialization in manufacturing, referenced in Section 4 of the paper and Supplementary Tables S6 and S7

- EU_US_connectivity_and_specialization.py: Script to compute the connectivity and specialization in manufacturing in the US and EU, referenced in Section 4 of the paper

#### Validation
Scripts to validate the data used in the Paper, referenced in the Supplementary Discussions S1--S3

- Crunchbase.ipynb: Script to validate the LinkedIn data with Crunchbase data. Note: Crunchbase data is not shared and needs to be downloaded separately (Supplementary Discussion S3)
- i3.ipynb: Script to validate the LinkedIn data with i3 data. Note: i3 data is not shared and needs to be downloaded separately (Supplementary Discussion S2)
- LinkedIn_Users_by_Country.ipynb: Script to validate the LinkedIn data with the number of LinkedIn users by country (Supplementary Figure S2)

