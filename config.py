
# parser = argparse.ArgumentParser(description="Create data for decompensation")
# parser.add_argument('--root_dir', type=str, help="Path to root folder containing all the patietns data")
# args, _ = parser.parse_known_args()

class Config():
    def __init__(self):
        self.seed = 36
        self.root_dir = '/media/ehealth/HDD/ICU/DataSets/eICU/Benchmark/pyscript/data1'
        self.eicu_dir = '/media/ehealth/HDD/ICU/DataSets/eICU'
        self.model_dir = ''
        self.num = True
        self.cat = True
        self.epochs = 100

        self.dec_cat = ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal']
        self.dec_num = ['admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)','Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
        'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']


        self.col_phe = ["Respiratory failure", "Fluid disorders",
                    "Septicemia", "Acute and unspecified renal failure", "Pneumonia",
                    "Acute cerebrovascular disease",
                    "Acute myocardial infarction", "Gastrointestinal hem", "Shock", "Pleurisy",
                    "lower respiratory", "Complications of surgical", "upper respiratory",
                    "Hypertension with complications", "Essential hypertension", "CKD", "COPD",
                    "lipid disorder", "Coronary athe", "DM without complication",
                    "Cardiac dysrhythmias",
                    "CHF", "DM with complications", "Other liver diseases", "Conduction disorders"]

        self.task = 'phen' #['phen', 'dec', 'mort', 'rlos']
     