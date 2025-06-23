import logging
import torch.nn as nn
from topolosses.losses.betti_matching import BettiMatchingLoss
from .dice import DiceLoss

logger = logging.getLogger(__name__)

# Registry of segmentation loss functions
loss_registry = {
    "dice": DiceLoss,
    "betti_matching": BettiMatchingLoss,
}

def get_segmentation_loss(config: dict) -> nn.Module:
    """
    Get the segmentation loss function based on the configuration.

    Args:
        config (dict): Configuration dictionary for the loss, 
                       e.g., {'name': 'dice', 'smooth': 1e-6}.

    Returns:
        nn.Module: The instantiated loss function.
    """
    loss_name = config.get("name", "dice") # Default to dice if not specified
    if not loss_name:
        logger.error("Loss function name is not specified in the configuration.")
        return None
    
    if loss_name not in loss_registry:
        logger.error(f"Loss function '{loss_name}' is not registered for segmentation. Available: {list(loss_registry.keys())}")
        return None
    
    loss_class = loss_registry[loss_name]
    logger.info(f"Using segmentation loss function: {loss_name}")

    # Extract additional parameters for the loss function from config
    loss_params = {k: v for k, v in config.items() if k != "name"}

    try:
        if loss_params:
            logger.info(f"Segmentation loss function parameters: {loss_params}")
            loss_instance = loss_class(**loss_params)
        else:
            loss_instance = loss_class()
        return loss_instance
    except Exception as e:
        logger.error(f"Error instantiating loss function {loss_name} with params {loss_params}: {e}")
        return None