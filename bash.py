from config import Config
from train import main
import os


TASK = ['mort', 'rlos', 'dec', 'phen']
NUM = [True, False]
CAT = [True, False]

class build_args():
    pass

for t in TASK:
    for n in NUM:
        for c in CAT:
            args = build_args()
            args.task = t
            args.num = n
            args.cat = c
            args.ohe = False
            args.ann = False
            args.mort_window = 24
            config = Config(args)
            print('Starting {}_num_{}_cat_{}'.format(t, str(n), str(c)))
            if not n and not c:
                continue
            output = main(config)

            if output:
                print('{}_num_{}_cat_{} finished.'.format(t, str(n), str(c)))
            else:
                print('error')
