import sys
from pathlib import Path
# add the parent of 'simulations/' (the repo root) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import importlib
from simulations.runtime_overrides import (
    add_runtime_override_args,
    normalize_config_module,
    prepare_runtime_config,
)

def semiparametrics_main():
    parser = argparse.ArgumentParser(description="Run semiparametric simulation sweeps.")
    parser.add_argument("--config", type=str, required=True, help="Config module/path.")
    add_runtime_override_args(parser)
    args = parser.parse_args(sys.argv[1:])

    config_module = normalize_config_module(args.config)
    config = importlib.import_module(config_module, __name__)
    runtime_config = prepare_runtime_config(config.CONFIG, args)
    from mcpy.SemiParametricsMonteCarlo import SemiParametricsMonteCarlo
    SemiParametricsMonteCarlo(runtime_config).run()
    
if __name__=="__main__":
    semiparametrics_main()
