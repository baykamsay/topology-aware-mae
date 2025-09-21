import logging
from functools import partial
from .mse import MaskedMSELoss
from .masked_betti_matching import BettiMatchingWithMSELoss

logger = logging.getLogger(__name__)

# Registry of loss functions
loss_registry = {
    "mse": MaskedMSELoss,
    "betti_matching": BettiMatchingWithMSELoss
}

def get_loss_function(config: dict, model) -> callable:
    """
    Get the loss function based on the configuration.

    Args:
        config (dict): Configuration dictionary containing the loss function name.

    Returns:
        callable: The loss function.
    """
    loss_name = config.get("name", "mse")
    if not loss_name:
        logger.error("Loss function name is not specified in the configuration.")
        return None
    
    if loss_name not in loss_registry:
        logger.error(f"Loss function '{loss_name}' is not registered. Available: {list(loss_registry.keys())}")
        return None
    
    loss_class = loss_registry[loss_name]
    logger.info(f"Using loss function: {loss_name}")

    # Extract additional parameters if needed
    loss_params = {k: v for k, v in config.items() if k != "name"}

    if loss_params:
        logger.info(f"Loss function parameters: {loss_params}")
        loss = loss_class(model, **loss_params)
    else:
        loss = loss_class(model)
    return loss