
# parser = argparse.ArgumentParser(description="Create data for decompensation")
# parser.add_argument('--root_dir', type=str, help="Path to root folder containing all the patietns data")
# args, _ = parser.parse_known_args()

class Config():
    def __init__(self, args):
        self.seed = 36

        # data dir
        self.root_dir = '/media/ehealth/HDD/ICU/DataSets/eICU/Benchmark/pyscript/data1'
        self.eicu_dir = '/media/ehealth/HDD/ICU/DataSets/eICU'

        # task details
        self.task = args.task #['phen', 'dec', 'mort', 'rlos']
        self.num = args.num #
        self.cat = args.cat #  
        self.n_cat_class = 429        

        self.k_fold = 2
        #model params
        self.model_dir = ''
        self.embedding_dim = 5
        self.epochs = 1
        self.batch_size = 512

        self.ann = args.ann #
        self.ohe = args.ohe #
        self.mort_window = args.mort_window #48 
        self.lr = 0.0001
        self.dropout = 0.3
        self.rnn_layers = 2
        self.rnn_units = [64, 64]


        # decompensation
        self.dec_cat = ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal']
        self.dec_num = ['admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)','Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
        'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']


        #phenotyping
        self.col_phe = ["Respiratory failure", "Fluid disorders",
                    "Septicemia", "Acute and unspecified renal failure", "Pneumonia",
                    "Acute cerebrovascular disease",
                    "Acute myocardial infarction", "Gastrointestinal hem", "Shock", "Pleurisy",
                    "lower respiratory", "Complications of surgical", "upper respiratory",
                    "Hypertension with complications", "Essential hypertension", "CKD", "COPD",
                    "lipid disorder", "Coronary athe", "DM without complication",
                    "Cardiac dysrhythmias",
                    "CHF", "DM with complications", "Other liver diseases", "Conduction disorders"]
     