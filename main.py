from runner import ExperimentRunner
from config import Config
import plot

if __name__ == '__main__':
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