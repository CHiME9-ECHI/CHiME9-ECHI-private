# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.inference.enhance import enhance_all_sessions
from scripts.inference.resample import resample_all_sessions


@hydra.main(version_base=None, config_path="config/inference", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.resample.run:
        resample_all_sessions(cfg.resample)

    if cfg.enhance.run:
        enhance_all_sessions(cfg.enhance)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
