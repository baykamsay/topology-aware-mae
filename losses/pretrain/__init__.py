import logging
from functools import partial
from .mse import mse_loss

logger = logging.getLogger(__name__)

# Registry of loss functions
loss_registry = {
    "mse": mse_loss,
}

def get_loss_function(config: dict) -> callable:
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
    
    loss_fn = loss_registry[loss_name]
    logger.info(f"Using loss function: {loss_name}")

    # Extract additional parameters if needed
    loss_params = {k: v for k, v in config.items() if k != "name"}

    if loss_params:
        logger.info(f"Loss function parameters: {loss_params}")
        return partial(loss_fn, **loss_params)
    else:
        return loss_fn