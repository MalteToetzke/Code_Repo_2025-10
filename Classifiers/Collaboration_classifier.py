from dotenv import dotenv_values
from openai import OpenAI
import pandas as pd
from typing import List
from sklearn.metrics import classification_report
import pickle
import re

class Collaboration_Classifier():

    def __init__(self):
        print("Initializing the Collaboration classifier")
        # load the CONFIG.env file which is in the root dir (two directories below)
        db_config = dotenv_values("CONFIG.env")

        self.client = OpenAI(api_key=db_config['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
        self.model = "deepseek-chat"

    def classify_announcements(self, texts: List[str]) -> List[dict]:
        """

        Args:
            texts: Linkedin_name: Post. referenced_organizations.

        Returns: List of dicts containing labels
        """

        self.set_prompt()

        # create an empty dataframe with the following columns:
        columns = ["post_id", "collaboration_id", "post_date", "collaborating_organizations",
                   "collaboration_relevance", "Biomass", "Biofuels", "Biogas", "Wind", "Offshore_Wind",
                   "Solar", "Concentrated_Solar", "Waste_to_Heat", "Direct_Air_Capture",
                   "Carbon_Capture_And_Storage", "Biochar", "BECCS",
                   "Carbon_Direct_Removal", "Hydrogen", "Nuclear_Energy", "Nuclear_Fusion", "Hydro_Energy",
                   "Geothermal", "Battery", "Electric_Vehicles", "Sustainable_Aviation_Fuels", "E_Fuels",
                   "Marine_Energy", "Heat_Pumps", "Railway", "Electric_Shipping", "Electric_Aviation",
                   "Fuel_Cell_Aviation",
                   "Other_Technology", "r_and_d_collaborations",
                   "demonstrations_and_pilots", "commercialisation_and_product_launches",
                   "production_and_manufacturing", "offtake_agreements_and_futures", "adoption_and_deployments",
                   "operations_and_maintenance", "grants", "equity_investments", "loans",
                   "other_unspecified_finance",
                   "spin_offs", "mergers_and_acquisitions", "joint_ventures", "incubators_and_accelerator_programs",
                   "certification_and_approvals", "training", "core_technology", "infrastructure",
                   "software_and_digital_platforms",
                   "raw_materials", "recycling", "other_complementary_activities"]
        df_results = pd.DataFrame(columns=columns)

        # create a second index for the additional rows in the dataframe
        data_frame_row = 0

        for i, text in enumerate(texts):
            # classify text

            #####
            prompt_template = f''' {self.prompt} \n  Post: {text} '''
            ######

            print(i)
            print(text)
            print("\n")

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=0, seed=1)

                answer = completion.choices[0].message.content
                print(answer)
            except Exception as e:
                print("Error during LLM call:")
                print(e)
                # Check if the error message contains 'Content Exists Risk'
                if 'Content Exists Risk' in str(e):
                    variable = 0  # Set the variable to 0
                    print("Error: Content Exists Risk detected. Variable set to 0.")
                    df_results.loc[data_frame_row, "collaboration_relevance"] = 0
                    data_frame_row += 1
                    continue

                else:
                    print("Error")
                    df_results.loc[data_frame_row, "collaboration_relevance"] = 0
                    data_frame_row += 1


            if "Yes" in str(answer[:15]):
                if "[Technologies" in str(answer):
                    print("-------- multiple collaborations ---------")

                    collaborations = answer.split("],[")
                    for iter, collaboration in enumerate(collaborations):
                        if iter>0:
                            print("FOR TEST REASONS WE ONLY TAKE THE FIRST COLLABORATION IDENTIFIED IN THE POST")
                            continue
                        df_results.loc[
                            data_frame_row, "collaborating_organizations"] = self.extract_collaborating_organizations(
                            collaboration)
                        df_results.loc[data_frame_row, "collaboration_relevance"] = 1
                        for column in columns[5:]:
                            if column in collaboration:

                                df_results.loc[data_frame_row, column] = 1
                            else:
                                df_results.loc[data_frame_row, column] = 0
                        data_frame_row += 1

                else:
                    df_results.loc[
                        data_frame_row, "collaborating_organizations"] = self.extract_collaborating_organizations(
                        answer)
                    df_results.loc[data_frame_row, "collaboration_relevance"] = 1
                    for column in columns[5:]:
                        if column in answer:
                            df_results.loc[data_frame_row, column] = 1
                        else:
                            df_results.loc[data_frame_row, column] = 0
                    data_frame_row += 1
            else:
                df_results.loc[data_frame_row, "collaboration_relevance"] = 0
                data_frame_row += 1

            print("------------------------------------------------")

        return df_results

    def set_prompt(self):
        self.prompt = """You are a research assistant classifying LinkedIn posts. Follow these steps:

                    1. Determine if the post announces an innovation collaboration between the posting organization and other organizations (Yes/No).
                       - Collaborations must explicitly involve a cooperation between entities for advancing a technology, OR a financial transaction, OR acquisitions, OR changes in ownership, OR any legally-binding partnership from one of the specified categories.
                       - Exclude joint events, panel discussions, or PR activities without specific innovation activities.

                    2. If Yes, classify:
                       - Technologies: Choose from the list ["Biomass", "Biofuels", "Biogas", "Wind", “Offshore_Wind”,"Solar", "Concentrated_Solar", "Waste_to_Heat", "Direct_Air_Capture", "Carbon_Capture_And_Storage", "Biochar", "BECCS", "Carbon_Direct_Removal", "Hydrogen" (including Fuel Cells for Green Energy), "Nuclear_Energy", "Nuclear_Fusion", "Hydro_Energy", "Geothermal", "Battery", "Electric_Vehicles", "Sustainable_Aviation_Fuels", "E_Fuels", "Marine_Energy" (tidal and wave electricity only!), "Heat_Pumps", "Railway", "Electric_Shipping", "Electric_Aviation", "Fuel_Cell_Aviation", "Other_Technology"].
                       - Collaboration types: Choose from the list { "r_and_d_collaborations": "Joint efforts to conduct research and develop new technologies.", "demonstrations_and_pilots": "Testing and demonstrating developed technologies to prove feasibility. Includes the deployment of prototype and testing facilities", "commercialisation_and_product_launches": “announcements of first commercial deployments and new product launches.", "production_and_manufacturing": "Setting up or scaling facilities to manufacture technologies or products.", "offtake_agreements_and_futures": "Contracts for future sales or delivery of products.", "adoption_and_deployments": " Concrete deployments, construction of power plants, sales, or adoption of products and technologies in the market. This includes the construction of infrastructure or systems that enable the use of a technology.", "operations_and_maintenance": "Collaborations focused on running and maintaining systems or equipment.", "grants": "Grants must be mentioned explicitely.", "equity_investments": "Financial investments in exchange for ownership stakes in a company or project.", "loans": "Explicitly mentioned debt or loan financing provided for project financing, repaid with interest.", "other_unspecified_finance": "Any financial collaboration that does not fit the categories above.", "spin_offs": "New companies formed to commercialize a specific technology or idea.", "mergers_and_acquisitions": "Business consolidations where companies merge or are acquired.", "joint_ventures": " Only classify as a joint venture if the term 'joint venture' is explicitly mentioned in the announcement!", "incubators_and_accelerator_programs": "Programs supporting startups or early-stage ventures through mentorship and resources.", “certification_and_approvals”:"Processes to obtain official recognition, regulatory approval, or standards compliance for technologies or products.”,"training":"training related to a technology provided from one organizations to another." }.
                       - Core technology: Whether the announcement is about the "core_technology" or corresponding "infrastructure", "software_and_digital_platforms", "raw materials", "recycling", or "other_complementary_activities". 

                    3. If Yes, extract the relevant organizations for the collaboration mentioned in the post from the list of 'referenced_organizations'.

                    If multiple distinct collaborations are announced in one post, separate them as a list.

                    Guidelines for Output: Use consistent formatting for responses. Only provide the classification. Do not repeat the post or add explanations. If multiple categories apply, include all relevant entries. In the innovation categories (["r_and_d_collaborations", "demonstrations_and_pilots", "commercialisation_and_product_launches", "production_and_manufacturing", "sales_agreements_and_futures") prioritize the most relevant. For financial collaborations, use “other_unspecified_finance” if the type of finance is unclear. Only classify technologies for tidal or wave electricity generation (and not shipping) as marine energy. Only classify airplanes and aviation projects with "Fuel_Cell_Aviation"! Do not change the format of the organization names in the referenced_organizations list.

                    Examples:
                    Post: fraunhofer-gesellschaft: We support Emmvee Group as their solar technology partner in setting up a production facility to produce solar panels at scale; referenced_organizations=['fraunhofer-gesellschaft','Emmvee Group']
                    Answer:Yes: Technologies = ['Solar'], Collaboration_types = ['production_and_manufacturing'], Core_technology = ['core_technology'], Collaborating_organizations:['fraunhofer-gesellschaft','Emmvee Group']}} 

                    Post: soato: Bill Nicholson presented a coating technology at CPhI. coating-society brix-technologies.
                    Answer: No: This is only a product announcement without specifying a collaboration.

                    Post: hydrogène-de-france: Milestones in Indonesia with u.s. international development finance corporation. Technical assistance to develop 22 hydrogen power plants. $1.5 billion potential investment; referenced_organizations=['hydrogène-de-france', 'us-idfc']
                    Answer: Yes: Technologies = ['Hydrogen'], Collaboration_types = ['other_unspecified_finance', 'offtake_agreements_and_futures']; Core_technology = ['core_technology'], Collaborating_organizations=['hydrogène-de-france', 'us-idfc']

                    Post: kengenkenya: KenGen Kenya has partnered with Nairobi Metropolitan Services (NMS) to develop a 45MW Waste to Heat Power Plant financed through a government grant; referenced_organizations= ['kengenkenya', 'kengenkenya']]
                    Answer: Yes: Technologies = ['Waste_to_Heat'], Collaboration_types = ['adoption_and_deployments','grants'], Core_technology = ['core_technology'], Collaborating_organizations=['kengenkenya', 'nairobi-metropolitan-services']

                    Post: vinci-concessions: since 2021, we are partenering with airbus to promote the use of hydrogen and accelerate the decarbonization of the aviation sector. In japan, kansai airports and airbushave recently signed a memorandum of understanding to explore the use of hydrogen at three airports we operate in the country. referenced_organizations=['vinci-concessions', 'airbus', 'kansai-airports']
                    Answer: Yes: Technologies = ['Hydrogen', 'Fuel_Cell_Aviation'], Collaboration_types = ['demonstrations_and_pilots'], Core_technology = ['core_technology'], Collaborating_organizations=['vinci-concessions', 'airbus', 'kansai-airports']

                    Post: volvo: Bosch at the World Business Council event highlighting joint efforts.
                    Answer: No: Participating in joint conferences and events does not count.

                    Post: faurencia: In partnership with our joint venture Symbio, we showcase tested prototypes for low and zero emission vehicles, including hydrogen and electric vehicles. Hydrogen News Hydrogen Daily; referenced_organizations=['faurencia', 'https://www.linkedin.com/company/symbio/', 'hydrogen-news', 'hydrogen-daily']
                    Answer: Yes: Technologies = ['Hydrogen', 'Electric_Vehicles']; Collaboration_types = ['joint_ventures', 'demonstrations_and_pilots'], Core_technology = ['core_technology'], Collaborating_organizations=['faurencia', 'https://www.linkedin.com/company/symbio/']

                    Post: Orbital Marine Power: The world's most powerful tidal turbine is installed, generating power into the UK grid. Made possible by European Commission and Horizon 2020 funding. referenced_organizations=['orbital-marine-power-ltd', 'european-commission', 'horizon-2020']
                    Answer: Yes: Technologies = ['Marine_Energy'], Collaboration_types = ['adoption_and_deployments', 'grants', 'equity_investments'], Core_technology = ['core_technology'], Collaborating_organizations=['orbital-marine-power-ltd', 'european-commission', 'horizon-2020']

                    Post: Amarenco: SolarPower Europe 𝗹𝗮𝘂𝗻𝗰𝗵𝗲𝘀 𝗙𝗿𝗲𝗻𝗰𝗵 𝘃𝗲𝗿𝘀𝗶𝗼𝗻 𝗼𝗳 𝘁𝗵𝗲 𝗔𝗴𝗿𝗶𝘀𝗼𝗹𝗮𝗿 𝗕𝗲𝘀𝘁 𝗣𝗿𝗮𝗰𝘁𝗶𝗰𝗲 𝗚𝘂𝗶𝗱𝗲𝗹𝗶𝗻𝗲𝘀, 𝘄𝗶𝘁𝗵 𝘁𝗵𝗲 𝘀𝘂𝗽𝗽𝗼𝗿𝘁 𝗼𝗳 𝗙𝗿𝗲𝗻𝗰𝗵 𝗶𝗻𝗱𝘂𝘀𝘁𝗿𝘆! The report explores real-world examples of Agrisolar projects in the EU. At Amarenco, we are extremely proud to be a part of this with SolarPower Europe; referenced_organizations=['SolarPower Europe', 'Amarenco', 'Amarenco']
                    Answer: No: This post is about the publication of a report.   

                    Post: AgriVijay: Presenting Multi Utility Biogas Powered Farming Tool power launched in #Partnershipwith Sukoon Solutions by AgriVijay ! Key benefits: Join the sustainable farming revolution today! Sukoon Solutions AIC-JKLU NITI  GOVERNMENT OF INDIA UNICEF  Confederation of India n Industry SIDBI(Small Industries Development Bank of India ) Social Alpha Omnivore. Here's a Glimpse of the product https://lnkd.in/dFs3uVPe; referenced_organizations= ['AgriVijay','sukoon-solutions', 'unicef', 'social-alpha', 'omnivore-partners']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                    Answer: Yes: Technologies = ['Biogas'], Collaboration_types = ['commercialisation_and_product_launches'], Core_technology = ['core_technology'], Collaborating_organizations=['AgriVijay', 'sukoon-solutions']

                    Post: siemens-energy: Today, we announce our new partnerships: with ABB we will develop a new generation of electric vehicle charging technologies and with SolarEdge we will construct a new solar farm in the US. referenced_organizations=['siemens-energy', 'abb', 'solaredge']
                    Answer: Yes: [Technologies = ['Electric_Vehicles'], Collaboration_types = ['r_and_d_collaborations'], Core_technology = ['infrastructure'], Collaborating_organizations=['siemens-energy', 'abb']],[Technologies = ['Solar'], Collaboration_types = ['production_and_manufacturing'], Core_technology = ['core_technology'], Collaborating_organizations:['siemens-energy', 'solaredge']

                    Post: grundon-waste-management-ltd: We’re delighted to announce we have just invested around £200,000 in two state-of-the-art JCB electric Teletruk forklifts by JBC. #electricvehicles #investment # JCB; referenced_organizations= ['grundon-waste-management-ltd', 'jcb', 'pyroban']
                    Answer: Yes: Technologies = ['Electric_Vehicles'], Collaboration_types = ['adoption_and_deployments'], Core_technology = ['core_technology'], Collaborating_organizations=['grundon-waste-management-ltd', 'jcb', 'pyroban']              

                    Post: clean-hydrogen-partnership: VIR TUAL-FCSannounces the 3rd release of its open-source platform for designing hybrid #fuelcell& battery systems developed in collaboration with Université Bourgogne Franche-Comté and Solaris Bus & Coach!Here's what the update will include; referenced_organizations= ['clean-hydrogen-partnership', 'virtual-fcs', 'sintef', 'universitebourgognefranchecomte', 'https://www.linkedin.com/company/solaris-bus-&-coach/']
                    Answer: Yes: Technologies = ['Hydrogen', 'Battery'], Collaboration_types = ['commercialisation_and_product_launches'], Core_technology = ['software_and_digital_platforms'], Collaborating_organizations=['virtual-fcs', 'universitebourgognefranchecomte', 'https://www.linkedin.com/company/solaris-bus-&-coach/']

                    Post: amerit-fleet-solutions: We are pleased to announce our agreement with The Shyft Group to provide mobile maintenance services to Blue Arc EVs fleet; referenced_organizations= ['amerit-fleet-solutions', 'The Shyft Group']
                    Answer: Yes: Technologies = ['Electric_Vehicles'], Collaboration_types = ['operations_and_maintenance'], Core_technology = ['core_technology'], Collaborating_organizations=['amerit-fleet-solutions', 'The Shyft Group']
                    """

    def extract_collaborating_organizations(self, input_string):
        match = re.search(r"Collaborating_organizations=\[([^\]]+)\]", input_string)

        # Extracting the matched string if it exists
        if match:
            collaborating_organizations_str = "[" + match.group(1) + "]"
            return collaborating_organizations_str
        else:
            match = re.search(r"Collaborating_organizations = \[([^\]]+)\]", input_string)
            if match:
                collaborating_organizations_str = "[" + match.group(1) + "]"
                return collaborating_organizations_str
            else:
                # raise an error
                raise ValueError("No collaborating organizations found in the input string")

    def text_processing(self, df):
        # text preprocessing
        df["referenced_organizations"] = "['" + df.Linkedin_name + "', " + df.referenced_organizations.str[1:] + "]"
        post_texts = list(
            df.Linkedin_name + ": " + df['text'] + "; referenced_organizations= " + df.referenced_organizations)

        return post_texts




if __name__ == "__main__":

    print(
    """
    Note: While temperature is set to 0 and a fixed seed is used, large language models (LLMs) are not fully deterministic. 
    Minor performance variations may occur between runs.

    Additionally, some LinkedIn posts contain multiple collaboration announcements. 
    Since results are retrieved ex post from the fully classified database, this can introduce small discrepancies 
    and may affect aggregate performance metrics.
    """)

    cl = Collaboration_Classifier()

    # RELEVANCE
    print("Classify Collaboration Relevance")
    df = pd.read_csv("Classifiers/Data/Collaboration_classifier/validation_relevance.csv")

    post_texts = cl.text_processing(df)

    results = cl.classify_announcements(post_texts)
    print(results)
    print(df.true_label)
    clf_report = classification_report(list(df['true_label']),list(results["collaboration_relevance"]))
    pickle.dump(clf_report, open('organization_type_classifier_classification_report.txt', 'wb'))
    print("Classification Report: ",clf_report)
    


    print("Use stored Predictions from Previous Runs")
    print("Classification report:")
    print(classification_report(df['true_label'], df['collaboration_classification']))
   
    # Technology
    print("Classify Technologies of Collaboration")
    df = pd.read_csv("Classifiers/Data/Collaboration_classifier/validation_technology.csv")
    label_cols = list(df.columns[1:-6])
    print(label_cols)

    post_texts = cl.text_processing(df)
    results = cl.classify_announcements(post_texts)

    # drop non-relevant rows
    ## get index of non-relevant
    non_relevant_indices = results[results['collaboration_relevance'] == 0].index
    ## drop non-relevant rows
    results = results.drop(non_relevant_indices)

    # only keep rows in df that are also in results
    df = df.drop(non_relevant_indices)
    print(df[label_cols].shape)
    print(results[label_cols].shape)
    clf_report = classification_report(df[label_cols].astype(int), results[label_cols].astype(int), target_names=label_cols)
    pickle.dump(clf_report, open('organization_type_classifier_classification_report_technologies.txt', 'wb'))
    print("Classification Report: ", clf_report)
  
    print("Import stored Predictions from Previous Runs")
    df_pred = pd.read_csv("Classifiers/Data/Collaboration_classifier/technology_predictions.csv")
    df = pd.read_csv("Classifiers/Data/Collaboration_classifier/validation_technology.csv")
    print("Classification report:")
    print(classification_report(df[label_cols].astype(int), df_pred[label_cols].astype(int), target_names=label_cols))

    # Collaboration Types
    print("Classify Collaboration Types")
    df = pd.read_csv("Classifiers/Data/Collaboration_classifier/validation_collaboration.csv")
    label_cols = list(df.columns[1:-6])
    print(label_cols)

    post_texts = cl.text_processing(df)
    results = cl.classify_announcements(post_texts)

    # drop non-relevant rows
    ## get index of non-relevant
    non_relevant_indices = results[results['collaboration_relevance'] == 0].index
    ## drop non-relevant rows
    results = results.drop(non_relevant_indices)

    # only keep rows in df that are also in results
    df = df.drop(non_relevant_indices)

    clf_report = classification_report(df[label_cols].astype(int), results[label_cols].astype(int), target_names=label_cols)
    pickle.dump(clf_report, open('organization_type_classifier_classification_report_collaboration_types.txt', 'wb'))
    print("Classification Report: ", clf_report)

    print("Import stored Predictions from Previous Runs")
    df = pd.read_csv("Classifiers/Data/Collaboration_classifier/validation_collaboration.csv")
    df_pred = pd.read_csv("Classifiers/Data/Collaboration_classifier/collaboration_predictions.csv")
    print("Classification report:")
    print(classification_report(df[label_cols].astype(int), df_pred[label_cols].astype(int), target_names=label_cols))
