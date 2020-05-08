from config import Config
from train import main
import os

from keras import backend as K


TASK = ['rlos']#, 'phen']#,'phen','rlos','dec']#['mort', 'rlos', 'phen','dec'] 
NUM = [True]
CAT = [True]
OHE = [False]
ANN = [False]

class build_args():
    pass

for t in TASK:
    for nu in NUM:
        for ca in CAT:
            for oh in OHE:
                for an in ANN:
                    K.clear_session()
                    args = build_args()
                    args.task = t
                    args.num = nu
                    args.cat = ca
                    args.ann = an
                    args.ohe = oh
                    # args.ohe = False
                    # args.ann = False
                    args.mort_window = 48
                    config = Config(args)
                    print('{}_num_{}_cat_{}_ohe_{} Started'.format(t, str(nu), str(ca),str(oh)))
                    if not nu and not ca:
                        print("do not do training without data ...")
                        continue
                    output = main(config)

                    if output:
                        print('{}_num_{}_cat_{}_ohe_{} Finished'.format(t, str(nu), str(ca),str(oh)))
                    else:
                        print('error')