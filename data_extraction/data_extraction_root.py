from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
from data_extraction import utils

def data_extraction_root(args):
    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    eicu_dir = os.path.join(args.eicu_dir)
    if not os.path.exists(eicu_dir):
        os.mkdir(eicu_dir)

    patients = utils.read_patients_table(args.eicu_dir, args.output_dir)
    stay_id = utils.cohort_stay_id(patients)
    utils.break_up_stays_by_unit_stay(patients, args.output_dir,stayid=stay_id, verbose=1)
    del patients

    # print("reading lab table")
    lab = utils.read_lab_table(args.eicu_dir)
    utils.break_up_lab_by_unit_stay(lab, args.output_dir, stayid=stay_id, verbose=1)
    del lab

    print("reading nurseCharting table, might take some time")
    nc = utils.read_nc_table(args.eicu_dir)
    utils.break_up_stays_by_unit_stay_nc(nc, args.output_dir, stayid=stay_id, verbose=1)
    del nc

    #Write the timeseries data into folders
    utils.extract_time_series_from_subject(args.output_dir)
    
    utils.delete_wo_timeseries(args.output_dir)
    #Write all the data into one dataframe
    utils.all_df_into_one_df(args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Create data for root")
    parser.add_argument('--eicu_dir', type=str, help="Path to root folder containing all the patietns data")
    parser.add_argument('--output_dir', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.eicu_dir):
        os.makedirs(args.eicu_dir)   
    data_extraction_root(args)
if __name__ == '__main__':
    main()