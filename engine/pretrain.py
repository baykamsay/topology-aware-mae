"""
Pretraining pipelines
"""

import math
import sys
from typing import Iterable

import torch

import utils.distributed as distributed
import utils.logging as logging
import utils.lr_schedule as lr_schedule

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    config: dict,
                    log_writer=None):
    model.train(True)
    metric_logger = logging.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', logging.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}/{config['training']['epochs']}]"
    print_freq = config['logging']['print_freq']

    update_freq = config['training'].get('update_freq', 1)
    num_training_steps_per_epoch = len(data_loader) // update_freq
    # log_config = config.get('logging', {})
    # enable_wandb = log_config.get('enable_wandb', False)

    optimizer.zero_grad()
    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = data_iter_step // update_freq
        global_step = epoch * num_training_steps_per_epoch + it
        # using per iteration in lr scheduler
        if data_iter_step % update_freq == 0:
            lr_schedule.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config['training'])
        
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss , _, _ = model(samples, labels, mask_ratio=config['model']['mask_ratio'])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        loss /= update_freq
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache() # clear gpu cache for large models
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = distributed.all_reduce_mean(loss_value)
        # if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
        #     """ Use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer(train_loss=loss_value_reduce, head="loss", step=epoch_1000x)
        #     log_data = {
        #         f"train/{k}": meter.global_avg
        #         for k, meter in metric_logger.meters.items()
        #     }
        #     log_data["train/step"] = global_step
        #     log_writer.log_epoch_metrics(log_data)

    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}