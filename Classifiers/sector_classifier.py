from dotenv import dotenv_values
from openai import OpenAI
import pandas as pd
from typing import List
from sklearn.metrics import classification_report
import pickle


class Sector:
    """
    Sector classifier using DeepSeek via the OpenAI interface.
    """

    def __init__(self):
        print("Initializing the OrgType classifier...")
        db_config = dotenv_values("CONFIG.env")
        self.client = OpenAI(api_key=db_config['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
        self.model = "deepseek-chat"

    def classify_sectors(self, texts: List[str]) -> List[str]:
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
        Initializes the base prompt with classification instructions and examples.
        """
        self.prompt = """
                 This is the data from organizations on linkedin. Classify the organization into one or multiple of the following sectors:
                             {
                     "Biomass": "Organic material used for energy production",
                     "Biofuels": "Liquid fuels from biological sources",
                     "Biogas": "Gas from anaerobic digestion of organic matter",
                     "Wind": "Energy from wind turbines",
                     "Solar": "Photovoltaic or solar thermal energy",
                     "Direct Air Capture": "Technology that removes CO₂ directly from air",
                     "Carbon Capture And Storage": "Capturing and storing CO₂ emissions",
                     "Other Carbon Removal": "Non-DAC, non-CCS carbon removal methods",
                     "Carbon Offsets": "Credits to compensate for CO₂ emissions",
                     "Hydrogen": "Hydrogen production and utilization for energy",
                     "Nuclear Energy": "Conventional nuclear fission energy",
                     "Nuclear Fusion": "Energy from fusion of atomic nuclei",
                     "Hydro energy": "Hydro power and hydro pumped storage",
                     "Geothermal": "Energy from Earth’s internal heat",
                     "Battery": "Electrochemical energy storage systems",
                     "Other Energy Storage": "Non-battery energy   storage (e.g., thermal)",
                     "Electricity-grid": "Power transmission and distribution infrastructure",
                     "Electric Vehicles": "Battery-powered vehicles",
                     "Sustainable Aviation Fuels": "Low-emission fuels for aircraft",
                     "E-Fuels": "Synthetic fuels from renewable electricity",
                     "Marine Energy": "Electricity from tidal or wave motion",
                     "Heat Pumps": "Devices for efficient heating and cooling",
                     "Other Renewables": "Renewable energy not otherwise listed",
                     "Gas": "Natural gas production or use",
                     "Oil": "Petroleum extraction, refining or distribution",
                     "Coal": "Coal mining and combustion",
                     "Other Fossil Energy": "Fossil-based energy not in gas/oil/coal",
                     "Energy General": "Broad or mixed energy sector activity",
                     "Railway": "Rail transport systems and services",
                     "Aviation": "Air transport and related infrastructure",
                     "Shipping": "Maritime transport of goods or people",
                     "Automotive": "Design or production of motor vehicles",
                     "Other Transport": "Transport modes not elsewhere specified",
                     "Chemical Industry": "Production of chemicals and related products",
                     "Built Environment": "Buildign constrction and operations",
                     "Other Construction": "Construction not focused on buildings",
                     "Computer and Electronic Products": "Hardware and electronic devices",
                     "Software": "Development of software products or services",
                     "Other ICT": "IT and communications not covered above   (e.g., cloud systems)",
                     "Agriculture": "Farming, crop, and livestock production",
                     "Iron and Steel": "Production of iron and steel materials",
                     "Other Manufacturing": "Manufacturing not in listed sectors",
                     "Military": "Defense and military-related activities",
                     "Raw Materials": "Mining or extraction of basic materials",
                     "Other Sectors": "Does not fit into listed sector categories"
                             }  

                 Guidelines for Output: Use consistent formatting for responses. Only provide the classification. Do not repeat the post or add explanations. Use the label "Other Sectors" if the organization does not specify any sector (e.g. financial services in general) or the sector does not fit to any of the labels (e.g., textiles). Marine energy refers to tidal and wave electricity generation only. 

                 Here are some exemplary classifications: 
                 Text: Fraunhofer-Institut für Solare Energiesysteme ISE; The Fraunhofer Institute for Solar Energy Systems ISE in Freiburg, Germany is the largest solar research institute in Europe. With a staff of about 1400, we are committed to promoting a sustainable, economic, secure and socially just energy supply system based on renewable energy sources. We contribute to this through our main research areas of energy provision, energy distribution, energy storage and energy utilization. Through outstanding research results, successful industrial projects, spin-off companies and global collaborations, we are shaping the sustainable transformation of the energy system.\n\nImprint: https://www.ise.fraunhofer.de/en/publishing-notes.html; ['Photovoltaik', 'Energiesystemtechnik', 'Solarthermie', 'Gebäudeenergietechnik', 'Wasserstofftechnologien']; ['#energytransition', '#research', '#solar']
                 Classification: Solar

                 Text: Esken; We own two core assets; London Southend Airport and Stobart Energy.\n\nLondon Southend Airport is an award-winning airport serving London and the South East. It has ambitions plans to grow and the key ingredients in place to enable that growth. And; in Stobart Energy we have a transformational and maturing business which supplies fuel to biomass plants across the UK so that they can, in turn, create renewable energy.; ['Infrastructure and Support Services', 'Railway Maintenance', 'Biomass Energy', 'Aviation', 'Biomass']; None
                 Classification: Aviation, Biomass, Railway

                 Text: LanzaTech; Where others see a dire choice, LanzaTech sees a trillion-dollar opportunity.\nThe good news is after 15 years, north of a thousand patents, \nand millions of hours of pioneering scientific inquiry, LanzaTech has invented a technology big enough to meet the moment. One that transforms pollution into profit, and ensures that  humans continue to prosper far into the post-pollution future.\nThe science is state-of-the-art, but the idea is simple. \nWe use nature to heal nature. First we capture carbon emissions. Then we feed them into bioreactors where trillions of carbon-hungry microbes go to work. These tiny dynamos eat the pollution and output valuable raw material commodities. Pure enough to be resold, repurposed, recycled, renewed, re-everythinged — from sustainable jet fuel to yoga pants.\n\nIt’s a commercial scale solution that’s ready for market today. In a crowded sector filled with speculation but short on results, our plug-and-play platform is already making our customers money. Turns out science and business makes one helluva team.\nWaste carbon pollution is humanity’s biggest threat. \nLanzaTech is turning it into an opportunity. Reducing emissions and making money for our customers — today. \nLet's transform our tired Lose-Lose climate debate into a Win-Win proposition. One that helps companies grow their revenue while helping the planet reach a post-pollution reality.\nLanzaTech | Welcome to the Post Pollution Future; ['chemical recycling', 'carbon emissions', 'air quality', 'circular economy', 'green chemistry', 'carbon recycling', 'climate action', 'fuels and chemicals']; ['#lanzatech', '#circulareconomy', '#carbonrecycling']
                 Classification: Other Carbon Removal, Sustainable Aviation Fuels, Chemical Industry 

                 Text: RWE; RWE is leading the way to a green energy world. With an extensive investment and growth strategy, the company will expand its powerful, green generation capacity to 50 gigawatts internationally by 2030. RWE is investing €50 billion gross for this purpose in this decade. The portfolio is based on offshore and onshore wind, solar, hydrogen, batteries, biomass and gas. RWE Supply & Trading provides tailored energy solutions for large customers. RWE has locations in the attractive markets of Europe, North America and the Asia-Pacific region. The company is responsibly phasing out nuclear energy and coal. Government-mandated phaseout roadmaps have been defined for both of these energy sources. RWE employs around 19,000 people worldwide and has a clear target: to get to net zero by 2040. On its way there, the company has set itself ambitious targets for all activities that cause greenhouse gas emissions. The Science Based Targets initiative has confirmed that these emission reduction targets are in line with the Paris Agreement. Very much in the spirit of the company’s purpose: Our energy for a sustainable life.; ['Power Generation', ' Renewable Energies', 'Gas Fleet', 'Energy Trading', 'Hydrogen']; ['#teamrwe']
                 Classification: Energy General

                 Text: Goldman Sachs; At Goldman Sachs, we believe progress is everyone’s business. That’s why we commit our people, capital and ideas to help our clients, shareholders and the communities we serve to grow.\nFounded in 1869, Goldman Sachs is a leading global investment banking, securities and investment management firm. Headquartered in New York, we maintain offices in all major financial centers around the world. \n\nMore about our company can be found at www.goldmansachs.com\n\nFor insights on developments currently shaping markets, industries and the global economy, subscribe to BRIEFINGS, a weekly email from Goldman Sachs. Copy and paste this link into your browser to sign up: http://link.gs.com/Qxf3\n\ngs.com/social-media-disclosures; []; ['#mynamemyheritage', '#makethingspossible']
                 Classification: Other Sectors

                 Text: PPC Solar (Paradise Power Company); PPC SOLAR excels in the engineering, procurement and construction (EPC) of energy transition solutions including photovoltaic installation, EV charging infrastructure and energy storage solutions. With over 40-years of local presence PPC has built strong relationships within the region for unsurpassed value. Headquartered in Taos, New Mexico, with a branch office in Albuquerque we serve the Southern, Central and Northern New Mexico Regions along
                 with Southern Colorado.; ['Photovoltaic', 'Solar', 'Renewable Energy', 'Solar Electric', 'PV ', 'EV Charging Stations', 'Energy Storage', 'battery based PV Systems', 'ChargePoint Partner', 'off-grid', 'Residential', 'Commercial', 'Utility', 'Operations & Maintenance', 'electric vehicle charging']; ['#newmexico', '#photovoltaics', '#evcharging']
                 Classification: Solar, Electric Vehicles, Battery

                 Text: PT Bukit Asam Tbk; PT Bukit Asam Tbk (IDX: PTBA) is a state-owned enterprise that operates in mining industry, particularly coal mining, located in Tanjung Enim, South Sumatra. In line with the Company’s vision to become a world-class energy company, Bukit Asam continues to be committed to increasing added value for the Company, shareholders, and all stakeholders of the Company.; ['Indonesia-State Owned Coal Mining Company', 'Coal Mining', 'Coal Trading', 'Energy']; ['#energi', '#sustainability', '#transformasidigital']
                 Classification: Coal

                 Text: Climeworks empowers people and companies to fight global warming by offering carbon dioxide removal as a service via direct air capture (DAC) technology. Climeworks’ DAC facilities run exclusively on clean energy, and our modular CO₂ collectors can be stacked to build machines of any capacity. 
                 Classification: Direct Air Capture

                 Text: Cisco; Cisco (NASDAQ: CSCO) enables people to make powerful connections--whether in business, education, philanthropy, or creativity. Cisco hardware, software, and service offerings are used to create the Internet solutions that make networks possible--providing easy access to information anywhere, at any time. Cisco was founded in 1984 by a small group of computer scientists from Stanford University. Since the company's inception, Cisco engineers have been leaders in the development of Internet Protocol (IP)-based networking technologies. Today, with more than 71,000 employees worldwide, this tradition of innovation continues with industry-leading products and solutions in the company's core development areas of routing and switching, as well as in advanced technologies such as home networking, IP telephony, optical networking, security, storage area networking, and wireless technology. In addition to its products, Cisco provides a broad range of service offerings, including technical support and advanced services. Cisco sells its products and services, both directly through its own sales force as well as through its channel partners, to large enterprises, commercial businesses, service providers, and consumers.; ['Networking', 'Wireless', 'Security', 'Unified Communication', 'Cloud', 'Collaboration', 'Data Center', 'Virtualization', 'Unified Computing Systems']; ['#lifeonwebex', '#wearecisco', '#bethebridge']
                 Classification: Computer and Electronic Products, Software, Other ICT

                 Text: Rio Tinto; We're finding better ways to provide the materials the world needs. Iron ore for steel. Low carbon aluminium for electric cars and smartphones. Copper for wind turbines, electric cars and the pipes that bring water to our home. Borates that help crops grow and titanium for paint.; ['mining', 'processing', 'exploration', 'metals', 'minerals', 'energy']
                 Classification: Iron and Steel, Raw Materials
                 """

if __name__ == "__main__":
    print(
        """
        Note: While temperature is set to 0 and a fixed seed is used, large language models (LLMs) are not fully deterministic. 
        Minor performance variations may occur between runs.
        """)
    cl = Sector()

    # Load labeled test dataset
    df = pd.read_csv("Classifiers/Data/sector_classifier_test_set.csv")
    label_cols = df.columns[5:].tolist()

    # Text preprocessing
    df["hashtags"] = df["hashtags"].fillna("")
    df["text"] = df["name"] + "; " + df["description"] + "; " + df["specialities"] + "; " + df["hashtags"]

    profiles = df["text"].tolist()

    # Predict sectors using DeepSeek/OpenAI
    results = cl.classify_sectors(profiles)
    df["answer"] = results
    df_pred = df[["text", "answer"]].copy()

    # Binary relevance transformation for multilabel evaluation
    for sector in label_cols:
        df_pred[sector] = df["answer"].str.contains(sector, na=False).astype(int)

    # Print and save classification report
    print(df[label_cols])
    print(df_pred[label_cols])
    # save the dataframe
    #df_pred.to_csv("Classifiers/Data/sector_classifier_test_set_predictions.csv", index=False)


    print("Classification report:")
    clf_report = classification_report(df[label_cols], df_pred[label_cols], target_names=label_cols)
    print(clf_report)
    pickle.dump(clf_report, open('sector_classifier_classification_report.txt', 'wb'))

    print("""Import stored Predictions from Previous Runs""")

    df_pred = pd.read_csv("Classifiers/Data/sector_classifier_test_set_predictions.csv")
    clf_report = pickle.load(open('organization_type_classifier_classification_report.txt', 'rb'))
    print("Classification report:")
    print(classification_report(df[label_cols], df_pred[label_cols], target_names=label_cols))
