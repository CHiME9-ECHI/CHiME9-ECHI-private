from omegaconf import DictConfig


def get_model(cfg: DictConfig, ckpt=None):
    """
    Get the model for the specified architecture.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model settings.

    Returns
    -------
    torch.nn.Module
        The model for the specified architecture.
    """

    if cfg.name == "causalmcxtfgridnet":
        from models.CausalMCxTFGridNet import MCxTFGridNet

        model = MCxTFGridNet(**cfg.params)
    else:
        raise ValueError(f"Unknown model name: {cfg.name}")

    if ckpt:
        model.load_state_dict(ckpt)

    return model
