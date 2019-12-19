from __future__ import absolute_import
from __future__ import print_function


import os
import numpy as np
import argparse
from data_extraction import utils
from config import Config

def data_extraction_decompensation(args):
    all_df = utils.embedding(args.root_dir)
    all_dec = utils.filter_decom_data(all_df)
    all_dec = utils.label_decompensation(all_dec)
    return all_dec

def main():
    config = Config()
    data = data_extraction_decompensation(config)

if __name__ == '__main__':
    main()