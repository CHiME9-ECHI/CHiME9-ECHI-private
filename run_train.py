# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.prepare import prepare
from scripts.train import run


@hydra.main(version_base=None, config_path="config", config_name="main_train")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.prepare_ref.run:
        prepare(cfg.prepare_ref)

    if cfg.prepare_train.run:
        prepare(cfg.prepare_train)

    if cfg.train.run:
        run(cfg.paths, cfg.model, cfg.train, True)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
