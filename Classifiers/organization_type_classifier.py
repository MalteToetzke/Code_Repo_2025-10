from dotenv import dotenv_values
from openai import OpenAI
import pandas as pd
from typing import List, Dict
from sklearn.metrics import classification_report
import pickle


class Org_Type:
    """
    Classifier for organization types using DeepSeek via the OpenAI interface.
    """

    def __init__(self):
        print("Initializing the OrgType classifier...")
        db_config = dotenv_values("CONFIG.env")
        self.client = OpenAI(api_key=db_config['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
        self.model = "deepseek-chat"

    def classify_org_types(self, texts: List[str]) -> List[str]:
        """
        Classifies a list of organization description texts into types.

        Args:
            texts (List[str]): List of textual descriptions of organizations.

        Returns:
            List[str]: List of classification labels returned by the model.
        """
        self.set_prompt()
        results = []

        for i, text in enumerate(texts):
            prompt_template = f"{self.prompt}\nText: {text}"
            print(f"[{i}] Submitting prompt for classification...")
            print(text, "\n")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_template}],
                temperature=0,
                seed=1
            )
            answer = completion.choices[0].message.content.strip().replace("Classification: ", "")
            print(f"Response: {answer}")
            print("------------------------------------------------")
            results.append(answer)

        return results

    def set_prompt(self):
        """
        Sets the prompt template with detailed classification instructions and examples.
        """
        self.prompt = """
              This is the data from organizations on linkedin. Classify the organization into one or multiple of the following categories:
              {
                "Governmental Organization": "Public sector entity or agency",
                "Non Governmental Organization": "Non-profit, non-state civil organization",
                "International Organization": "Multinational governance or policy body",
                "Research Institute": "Non-university research body",
                "University": "Higher education and academic research institution",
                "Accelerator/Incubator": "Startup support program or ecosystem",
                "Bank": "Financial institution offering banking services",
                "Venture Capital": "Investment firm funding startups and innovation",
                "Other Financing Organization": "Non-bank financial services (e.g., real estate funds, private equity, pensin funds)",
                "Consulting": "Professional services firm offering expert advice",
                "Project Developer": "Company developing infrastructure or energy projects",
                "Manufacturer": "Entity producing physical goods or equipment",
                "Mining": "Entity involved in mineral or resource extraction",
                "Utility": "Company providing electricity, water, or gas services",
                "Service Provider": "Commercial firm delivering operational or technical services",
                "Lobby Group": "Organization advocating for specific policy interests",
                "Media Firm": "News or information dissemination organization",
                "Event Organizer": "Entity managing or hosting conferences and events",
                "Other": "Does not fit into any defined categories"
                }
              Use the label "Other" if the organization does not fit into any of the above categories.  Only provide the classification. Do not repeat the post or add explanations.

              Here are some exemplary classifications: 
              Text: Fraunhofer-Institut für Solare Energiesysteme ISE; The Fraunhofer Institute for Solar Energy Systems ISE in Freiburg, Germany is the largest solar research institute in Europe. With a staff of about 1400, we are committed to promoting a sustainable, economic, secure and socially just energy supply system based on renewable energy sources. We contribute to this through our main research areas of energy provision, energy distribution, energy storage and energy utilization. Through outstanding research results, successful industrial projects, spin-off companies and global collaborations, we are shaping the sustainable transformation of the energy system.\n\nImprint: https://www.ise.fraunhofer.de/en/publishing-notes.html; ['Photovoltaik', 'Energiesystemtechnik', 'Solarthermie', 'Gebäudeenergietechnik', 'Wasserstofftechnologien']; ['#energytransition', '#research', '#solar']
              Classification: Research Institute

              Text: Esken; We own two core assets; London Southend Airport and Stobart Energy.\n\nLondon Southend Airport is an award-winning airport serving London and the South East. It has ambitions plans to grow and the key ingredients in place to enable that growth. And; in Stobart Energy we have a transformational and maturing business which supplies fuel to biomass plants across the UK so that they can, in turn, create renewable energy.; ['Infrastructure and Support Services', 'Railway Maintenance', 'Biomass Energy', 'Aviation', 'Biomass']; None
              Classification: Service Organization

              Text: LanzaTech; Where others see a dire choice, LanzaTech sees a trillion-dollar opportunity.\nThe good news is after 15 years, north of a thousand patents, \nand millions of hours of pioneering scientific inquiry, LanzaTech has invented a technology big enough to meet the moment. One that transforms pollution into profit, and ensures that  humans continue to prosper far into the post-pollution future.\nThe science is state-of-the-art, but the idea is simple. \nWe use nature to heal nature. First we capture carbon emissions. Then we feed them into bioreactors where trillions of carbon-hungry microbes go to work. These tiny dynamos eat the pollution and output valuable raw material commodities. Pure enough to be resold, repurposed, recycled, renewed, re-everythinged — from sustainable jet fuel to yoga pants.\n\nIt’s a commercial scale solution that’s ready for market today. In a crowded sector filled with speculation but short on results, our plug-and-play platform is already making our customers money. Turns out science and business makes one helluva team.\nWaste carbon pollution is humanity’s biggest threat. \nLanzaTech is turning it into an opportunity. Reducing emissions and making money for our customers — today. \nLet's transform our tired Lose-Lose climate debate into a Win-Win proposition. One that helps companies grow their revenue while helping the planet reach a post-pollution reality.\nLanzaTech | Welcome to the Post Pollution Future; ['chemical recycling', 'carbon emissions', 'air quality', 'circular economy', 'green chemistry', 'carbon recycling', 'climate action', 'fuels and chemicals']; ['#lanzatech', '#circulareconomy', '#carbonrecycling']
              Classification: Manufacturer

              Text: RWE; RWE is leading the way to a green energy world. With an extensive investment and growth strategy, the company will expand its powerful, green generation capacity to 50 gigawatts internationally by 2030. RWE is investing €50 billion gross for this purpose in this decade. The portfolio is based on offshore and onshore wind, solar, hydrogen, batteries, biomass and gas. RWE Supply & Trading provides tailored energy solutions for large customers. RWE has locations in the attractive markets of Europe, North America and the Asia-Pacific region. The company is responsibly phasing out nuclear energy and coal. Government-mandated phaseout roadmaps have been defined for both of these energy sources. RWE employs around 19,000 people worldwide and has a clear target: to get to net zero by 2040. On its way there, the company has set itself ambitious targets for all activities that cause greenhouse gas emissions. The Science Based Targets initiative has confirmed that these emission reduction targets are in line with the Paris Agreement. Very much in the spirit of the company’s purpose: Our energy for a sustainable life.; ['Power Generation', ' Renewable Energies', 'Gas Fleet', 'Energy Trading', 'Hydrogen']; ['#teamrwe']
              Classification: Utility

              Text: Goldman Sachs; At Goldman Sachs, we believe progress is everyone’s business. That’s why we commit our people, capital and ideas to help our clients, shareholders and the communities we serve to grow.\nFounded in 1869, Goldman Sachs is a leading global investment banking, securities and investment management firm. Headquartered in New York, we maintain offices in all major financial centers around the world. \n\nMore about our company can be found at www.goldmansachs.com\n\nFor insights on developments currently shaping markets, industries and the global economy, subscribe to BRIEFINGS, a weekly email from Goldman Sachs. Copy and paste this link into your browser to sign up: http://link.gs.com/Qxf3\n\ngs.com/social-media-disclosures; []; ['#mynamemyheritage', '#makethingspossible']
              Classification: Bank

              Text: PPC Solar (Paradise Power Company); PPC SOLAR excels in the engineering, procurement and construction (EPC) of energy transition solutions including photovoltaic installation, EV charging infrastructure and energy storage solutions. With over 40-years of local presence PPC has built strong relationships within the region for unsurpassed value. Headquartered in Taos, New Mexico, with a branch office in Albuquerque we serve the Southern, Central and Northern New Mexico Regions along
              with Southern Colorado.; ['Photovoltaic', 'Solar', 'Renewable Energy', 'Solar Electric', 'PV ', 'EV Charging Stations', 'Energy Storage', 'battery based PV Systems', 'ChargePoint Partner', 'off-grid', 'Residential', 'Commercial', 'Utility', 'Operations & Maintenance', 'electric vehicle charging']; ['#newmexico', '#photovoltaics', '#evcharging']
              Classification: Manufacturer, Project Developer

              Text: The ERA Foundation; The ERA Foundation is a non-profit organisation which supports engineering skills development and aims to bridge the gap between engineering research and its commercialisation.; ['Outreach', 'Policy', 'Supporting UK Manufacturing', 'Supporting Engineering in the UK']; ['#engineering', '#stemeducation', '#electech']
              Classification: Non Governmental Organization

              Text: Parkit Enterprise Inc.; Parkit Enterprise Inc. is engaged in the acquisition, optimization and asset management of income producing industrial properties in Canada and parking facilities across North America. The Company's shares are listed on TSX-V (Symbol: PKT) and on the OTC (Symbol: PKTEF).; ['Industrial Real Estate', 'Parking Assets']; ['#parkitenterprise']
              Classification: Other Financing Organization

              Text: PT Bukit Asam Tbk; PT Bukit Asam Tbk (IDX: PTBA) is a state-owned enterprise that operates in mining industry, particularly coal mining, located in Tanjung Enim, South Sumatra. 

              In line with the Company’s vision to become a world-class energy company, Bukit Asam continues to be committed to increasing added value for the Company, shareholders, and all stakeholders of the Company.; ['Indonesia-State Owned Coal Mining Company', 'Coal Mining', 'Coal Trading', 'Energy']; ['#energi', '#sustainability', '#transformasidigital']
              Classification: Governmental Organization, Mining

              Text: The Economist; News and analysis with a global perspective. We’re here to help you understand the world around you. To subscribe to The Economist go to: https://bit.ly/3OLFYsJ To make a request about your personal data visit: https://econ.st/2OZnq92; ['News', 'Analysis', 'Global Perspective']; ['#climatechange', '#sustainability', '#environment']
              Classification: Media Firm

              """

if __name__ == "__main__":
    print(
        """
        Note: While temperature is set to 0 and a fixed seed is used, large language models (LLMs) are not fully deterministic. 
        Minor performance variations may occur between runs.
        """)

    cl = Org_Type()

    # Load labeled test set
    df = pd.read_csv("Classifiers/Data/organization_type_classifier_test_set.csv")
    label_cols = df.columns[5:].tolist()
    print(label_cols)

    # Fill missing hashtags and construct combined input text
    df["hashtags"] = df["hashtags"].fillna("")
    df["text"] = df["name"] + "; " + df["description"] + "; " + df["specialities"] + "; " + df["hashtags"]
    profiles = df["text"].tolist()

    # Classify using DeepSeek
    results = cl.classify_org_types(profiles)
    df["answer"] = results
    df_pred = df[["text", "answer"]].copy()

    # Binary relevance assignment for multi-label classification
    for firm_type in label_cols:
        if firm_type == "Governmental Organization":
            print("Governmental Organization")
            df_pred[firm_type] = df["answer"].str.contains("(?<!Non\s)Governmental Organization", regex=True, na=False, case=False).astype(int)
        else:
            df_pred[firm_type] = df["answer"].str.contains(firm_type, na=False, case=False).astype(int)

    # Evaluation report
    print(df[label_cols])
    print(df_pred[label_cols])
    #save the outcomes
    #df_pred.to_csv("Classifiers/Data/organization_type_classifier_test_set_predictions.csv", index=False)
    

    # Save classification report
    print("Classification report:")
    print(classification_report(df[label_cols], df_pred[label_cols], target_names=label_cols))

    clf_report = classification_report(df[label_cols], df_pred[label_cols], target_names=label_cols)
    pickle.dump(clf_report, open('organization_type_classifier_classification_report.txt', 'wb'))


    print("""Import stored Predictions from Previous Runs""")

    df_pred = pd.read_csv("Classifiers/Data/organization_type_classifier_test_set_predictions.csv")
    clf_report = pickle.load(open('organization_type_classifier_classification_report.txt', 'rb'))
    print("Classification report:")
    print(classification_report(df[label_cols], df_pred[label_cols], target_names=label_cols))