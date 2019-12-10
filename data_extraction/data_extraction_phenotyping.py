from __future__ import absolute_import
from __future__ import print_function


import os
import numpy as np
import argparse
from data_extraction import utils
from config import Config
import json
def data_extraction_phenotyping(args):
    label_pheno = ['Respiratory failure', 'Essential hypertension',
                'Cardiac dysrhythmias', 'Fluid disorders', 'Septicemia',
                'Acute and unspecified renal failure', 'Pneumonia',
                'Acute cerebrovascular disease', 'CHF', 'CKD', 'COPD',
                'Acute myocardial infarction', "Gastrointestinal hem",
                'Shock', 'lipid disorder', 'DM with complications', 'Coronary athe',
                'Pleurisy', 'Other liver diseases', 'lower respiratory',
                'Hypertension with complications', 'Conduction disorders',
                'Complications of surgical', 'upper respiratory',
                'DM without complication']

    diag_ord_col = ["patientunitstayid", "itemoffset", "Respiratory failure", "Fluid disorders",
                    "Septicemia", "Acute and unspecified renal failure", "Pneumonia",
                    "Acute cerebrovascular disease",
                    "Acute myocardial infarction", "Gastrointestinal hem", "Shock", "Pleurisy",
                    "lower respiratory", "Complications of surgical", "upper respiratory",
                    "Hypertension with complications", "Essential hypertension", "CKD", "COPD",
                    "lipid disorder", "Coronary athe", "DM without complication",
                    "Cardiac dysrhythmias",
                    "CHF", "DM with complications", "Other liver diseases", "Conduction disorders"]
    
    diag_columns = ['patientunitstayid', 'itemoffset',
                'Respiratory failure',
                'Essential hypertension', 'Cardiac dysrhythmias',
                'Fluid disorders', 'Septicemia',
                'Acute and unspecified renal failure', 'Pneumonia',
                'Acute cerebrovascular disease', 'CHF', 'CKD', 'COPD',
                'Acute myocardial infarction', "Gastrointestinal hem",
                'Shock', 'lipid disorder', 'DM with complications', 'Coronary athe',
                'Pleurisy', 'Other liver diseases', 'lower respiratory',
                'Hypertension with complications', 'Conduction disorders',
                'Complications of surgical', 'upper respiratory',
                'DM without complication']

    codes = json.load(open('phen_code.json'))
    all_df = utils.embedding(args.root_dir)
    all_df = utils.filter_phenotyping_data(all_df)
    diag = utils.read_diagnosis_table(args.eicu_dir)
    diag = utils.diag_labels(diag)

    diag.dropna(how='all', subset=label_pheno, inplace=True)
    stay_diag = set(diag['patientunitstayid'].unique())
    stay_all = set(all_df.patientunitstayid.unique())
    stay_intersection = stay_all.intersection(stay_diag)
    stay_pheno = list(stay_intersection)

    diag = diag[diag['patientunitstayid'].isin(stay_pheno)]
    diag.rename(index=str, columns={"diagnosisoffset": "itemoffset"}, inplace=True)
    diag = diag[diag_columns]
    label = diag.groupby('patientunitstayid').sum()
    label = label.reset_index()
    label[label_pheno] = np.where(label[label_pheno] >= 1, 1, label[label_pheno])
    all_pheno = all_df[all_df["patientunitstayid"].isin(stay_pheno)]
    all_pheno = all_pheno[all_pheno["itemoffset"] > 0]  # remove records before unit admission
    all_pheno = all_pheno[all_pheno["RLOS"] >= 0]  # remove records after unit discharge
    
    label = label[diag_ord_col]
    all_pheno_label = label[label.patientunitstayid.isin(list(all_pheno.patientunitstayid.unique()))]
    return all_pheno, all_pheno_label


def main():
    config = Config()
    data = data_extraction_phenotyping(config)

if __name__ == '__main__':
    main()