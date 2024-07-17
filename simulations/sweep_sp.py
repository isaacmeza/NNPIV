import sys
import argparse
from mcpy.SemiParametricsMonteCarlo import SemiParametricsMonteCarlo
import importlib

def semiparametrics_main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args(sys.argv[1:])

    config = importlib.import_module(args.config, __name__)
    SemiParametricsMonteCarlo(config.CONFIG).run()
    
if __name__=="__main__":
    semiparametrics_main()