from runner import ExperimentRunner
from config import Config
import plot
import random
import numpy as np
import torch

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # get parsed arguments
    cfg = Config().get_config()

    # init env
    runner = ExperimentRunner(cfg)

    # run DRL agents
    if cfg.train:
        runner.train()
    if cfg.evaluate:
        runner.evaluate()

    # run heuristic
    runner.heuristic()

    # plot results
    plot.show_results(runner)