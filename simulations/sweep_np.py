import sys
import argparse
from simulations.mcpy.NonParametricsMonteCarlo import NonParametricsMonteCarlo
import importlib

def monte_carlo_main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args(sys.argv[1:])

    config = importlib.import_module(args.config, __name__)
    NonParametricsMonteCarlo(config.CONFIG).run()
    
if __name__=="__main__":
    monte_carlo_main()