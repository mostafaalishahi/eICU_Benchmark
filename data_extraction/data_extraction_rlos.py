from __future__ import absolute_import
from __future__ import print_function


import os
import numpy as np
import argparse
from data_extraction import utils
from config import Config


def data_extraction_rlos(args):
    all_df = utils.embedding(args.root_dir)
    all_los = utils.filter_rlos_data(all_df)

    return all_los



def main():
    config = Config()
    data = data_extraction_rlos(config)

if __name__ == '__main__':
    main()