import os
import argparse
import yaml
import script.train as train
import script.test as test
import script.real as real
from util.data_util import set_random_seed


parser = argparse.ArgumentParser()

parser.add_argument('--yaml', type=str, default='*.yaml')
args = parser.parse_args()


option = yaml.safe_load(open(args.yaml))
set_random_seed(seed=option['global_setting']['seed'])
os.environ['CUDA_VISIBLE_DEVICES'] = option['global_setting']['device']


if __name__ == '__main__':
    if option['global_setting']['action'].upper() == 'train'.upper():
        train.inlet(option=option, args=args)
    elif option['global_setting']['action'].upper() == 'test'.upper():
        test.inlet(option, args=args)
    else:
        real.inlet(option, args=args)
