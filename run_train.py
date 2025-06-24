# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.setup import setup
from scripts.train import run


@hydra.main(version_base=None, config_path="config", config_name="main_train")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.setup_train_targets.run:
        setup(cfg.setup_train_targets)

    if cfg.setup_train_inputs.run:
        setup(cfg.setup_train_inputs)

    if cfg.train.run:
        run(cfg.paths, cfg.model, cfg.train, True)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
