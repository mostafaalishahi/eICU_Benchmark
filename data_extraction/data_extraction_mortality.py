from __future__ import absolute_import
from __future__ import print_function


import os
import numpy as np
import argparse
from data_extraction import utils
from config import Config

import pandas as pd
def data_extraction_mortality(args):
    time_window = args.time_window
    all_df = utils.embedding(args.root_dir)
    all_mort = utils.filter_mortality_data(all_df)
    all_mort = all_mort[all_mort['itemoffset']<=time_window]
    return all_mort

def main():
    config = Config()
    data = data_extraction_mortality(config)

if __name__ == '__main__':
    main()
    