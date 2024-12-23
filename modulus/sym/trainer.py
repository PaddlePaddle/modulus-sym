# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Modulus Solver
"""

import os
import time
import numpy as np
import paddle
from tensorboardX import SummaryWriter
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler
from paddle.amp import GradScaler
import datetime
from paddle import nn
from paddle import profiler
import paddle.distributed as dist
from termcolor import colored, cprint
from copy import copy
from operator import add
from omegaconf import DictConfig, OmegaConf
import hydra
import itertools
from collections import Counter
from typing import Dict, List, Optional
import logging
from contextlib import ExitStack
from contextlib import nullcontext

from .domain.constraint import Constraint
from .domain import Domain
from .loss.aggregator import Sum
from .utils.training.stop_criterion import StopCriterion
from .constants import TF_SUMMARY
from .hydra import (
    instantiate_optim,
    instantiate_agg,
    add_hydra_run_path,
)
from .distributed.manager import DistributedManager

from contextlib import ContextDecorator


class AdamMixin:
    """Special functions for training using the standard optimizers
    Should be used with ADAM, SGD, RMSProp, etc.
    """

    def adam_compute_gradients(
        self, aggregator: nn.Layer, global_optimizer_model: nn.Layer, step: int
    ):
        loss, losses = 0, Counter({})
        for agg_step in range(self.grad_agg_freq):
            with self.auto_cast_ctx:
                paddle.framework.core.nvprof_nvtx_push("Loss computation")
                losses_minibatch = self.compute_losses(step)
                paddle.framework.core.nvprof_nvtx_pop()
                if self.grad_agg_freq > 1:
                    losses_minibatch = {
                        key: value / self.grad_agg_freq
                        for key, value in losses_minibatch.items()
                    }
                paddle.framework.core.nvprof_nvtx_push("Loss aggregator")
                loss_minibatch = aggregator(losses_minibatch, step)
                paddle.framework.core.nvprof_nvtx_pop()
                loss += loss_minibatch
            paddle.framework.core.nvprof_nvtx_push("Weight gradients")
            if not self.enable_scaler:
                loss_minibatch.backward()
            else:
                self.scaler.scale(loss_minibatch).backward()
            paddle.framework.core.nvprof_nvtx_pop()
            losses.update(losses_minibatch)

        return loss, dict(losses)

    def adam_apply_gradients(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()


class AdaHessianMixin:
    """Special functions for training using the higher-order optimizer AdaHessian"""

    def adahess_compute_gradients(
        self, aggregator: nn.Layer, global_optimizer_model: nn.Layer, step: int
    ):
        if self.amp:
            raise NotImplementedError("AMP is not supported for this optimizer.")
        # With data hessian we need to keep grad graph on back-prop to approximate
        # the hessian with. The suggested Paddle way is to use paddle.grad instead
        # of backward.
        loss, losses = 0, Counter({})
        grads = [
            paddle.zeros_like(parameter)
            for parameter in list(global_optimizer_model.parameters())
        ]
        for agg_step in range(self.grad_agg_freq):
            losses_minibatch = self.compute_losses(step)
            losses_minibatch = {
                key: value / self.grad_agg_freq
                for key, value in losses_minibatch.items()
            }
            loss_minibatch = aggregator(losses_minibatch, step)

            grads_step = paddle.grad(
                loss_minibatch,
                list(global_optimizer_model.parameters()),
                create_graph=True,
            )
            grads = list(map(add, grads, grads_step))

            loss += loss_minibatch
            losses.update(losses_minibatch)
        # Set gradients of models manually
        for grad, param in zip(grads, global_optimizer_model.parameters()):
            param.grad = grad

        return loss, dict(losses)

    def adahess_apply_gradients(self):
        self.adam_apply_gradients()


class BFGSMixin:
    """Special functions for training using BFGS optimizer"""

    def bfgs_compute_gradients(
        self, aggregator: nn.Layer, global_optimizer_model: nn.Layer, step: int
    ):
        # Dummy functioned used entirely just for logging purposes and storing
        # objects for internal BFGS updates. Gradients are not calc'd here for BFGS
        if self.amp:
            raise NotImplementedError("AMP is not supported for this optimizer.")
        if self.max_steps != 0:
            self.log.warning("lbfgs optimizer selected. Setting max_steps to 0")
            self.max_steps = 0
        if self.grad_agg_freq != 1:
            self.log.warning("lbfgs optimizer selected. Setting grad_agg_freq to 1")
            self.grad_agg_freq = 1
        losses = self.compute_losses(step)
        loss = aggregator(losses, step)

        self.bfgs_step = step
        self.bfgs_aggregator = aggregator
        # Re-zero any gradients
        for param in global_optimizer_model.parameters():
            param.grad = None

        return loss, losses

    def bfgs_closure_func(self):
        self.optimizer.clear_grad()
        loss = 0
        losses = self.compute_losses(self.bfgs_step)
        loss = self.bfgs_aggregator(losses, self.bfgs_step)

        loss.backward()
        self.bfgs_optim_steps += 1
        return loss

    def bfgs_apply_gradients(self):
        assert (
            not self.bfgs_aggregator is None
        ), "Call bfgs_compute_gradients prior to this!"
        assert not self.bfgs_step is None, "Call bfgs_compute_gradients prior to this!"
        self.bfgs_optim_steps = 0
        self.log.info(f"[step: {self.bfgs_step:10d}] lbfgs optimization in running")
        self.optimizer.step(self.bfgs_closure_func)
        self.log.info(
            f"lbfgs optimization completed after {self.bfgs_optim_steps} steps"
        )


# base class for optimizing networks on loss
class Trainer(AdamMixin, AdaHessianMixin, BFGSMixin):
    """Base class for optimizing networks on losses/constraints"""

    def __init__(self, cfg: DictConfig):
        super(Trainer, self).__init__()

        # Save a local copy of the config
        self.cfg = cfg

        # set training parameters
        self._network_dir = self.cfg.network_dir
        self._initialization_network_dir = self.cfg.initialization_network_dir
        self.max_steps = self.cfg.training.max_steps
        self.grad_agg_freq = self.cfg.training.grad_agg_freq
        self.save_network_freq = self.cfg.training.save_network_freq
        self.print_stats_freq = self.cfg.training.print_stats_freq
        self.summary_freq = self.cfg.training.summary_freq
        self.amp = self.cfg.training.amp
        self.stop_criterion_metric = self.cfg.stop_criterion.metric
        self.stop_criterion_min_delta = self.cfg.stop_criterion.min_delta
        self.stop_criterion_patience = self.cfg.stop_criterion.patience
        self.stop_criterion_mode = self.cfg.stop_criterion.mode
        self.stop_criterion_freq = self.cfg.stop_criterion.freq
        self.stop_criterion_strict = self.cfg.stop_criterion.strict

        self.save_filetypes = self.cfg.save_filetypes
        self.summary_histograms = self.cfg.summary_histograms

        self.apply_gradients = self._apply_gradients
        self.compute_gradients = self._compute_gradients

        # make logger
        self.log = logging.getLogger(__name__)

        # Set distributed manager
        self.manager = DistributedManager()

        # set device
        self.place = self.manager.place
        self.device_amp = "cuda" if self.manager.cuda else "cpu"

        # set amp dtype
        if self.cfg.training.amp_dtype == "bfloat16" or self.device_amp == "cpu":
            self.amp_dtype = "bfloat16"
            if self.device_amp == "cpu" and self.amp:
                self.log.warning(
                    "Switching amp_dtype to bfloat16, AutocastCPU only supports bfloat16"
                )
        else:
            self.amp_dtype = "float16"

    def compute_losses(self, step: int):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def _compute_gradients(self):
        raise NotImplementedError("Config should set the compute_gradients function")

    def _apply_gradients(self):
        raise NotImplementedError("Config should set the apply_gradients function")

    def get_saveable_models(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def create_global_optimizer_model(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def load_network(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def save_checkpoint(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_constraints(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_validators(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    @property
    def has_validators(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_inferencers(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    @property
    def has_inferencers(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_monitors(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    @property
    def has_monitors(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def get_num_losses(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def _record_constraints(self):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            rec_inferencer_start = time.perf_counter()
            self.record_constraints()
            self.log.debug(
                f"{self.step_str} saved constraint results to {self.network_dir}"
            )
            self.log.info(
                f"{self.step_str} record constraint batch time: {time.perf_counter()-rec_inferencer_start:10.3e}s"
            )

    def _record_validators(self, step):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            rec_validation_start = time.perf_counter()
            self.validator_outvar = self.record_validators(step)
            self.log.debug(
                f"{self.step_str} saved validator results to {self.network_dir}"
            )
            self.log.info(
                f"{self.step_str} record validators time: {time.perf_counter()-rec_validation_start:10.3e}s"
            )

    def _record_inferencers(self, step):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            rec_inferencer_start = time.perf_counter()
            self.record_inferencers(step)
            self.log.debug(
                f"{self.step_str} saved inferencer results to {self.network_dir}"
            )
            self.log.info(
                f"{self.step_str} record inferencers time: {time.perf_counter()-rec_inferencer_start:10.3e}s"
            )

    def _record_monitors(self, step):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            rec_monitor_start = time.perf_counter()
            self.monitor_outvar = self.record_monitors(step)
            self.log.debug(
                f"{self.step_str} saved monitor results to {self.network_dir}"
            )

            # write parameter histograms to tensorboard
            if self.summary_histograms:
                for (
                    name,
                    parameter,
                ) in self.global_optimizer_model.named_parameters():
                    name = name.split(".")
                    name = ".".join(name[:-1]) + "/" + ".".join(name[-1:])
                    self.writer.add_histogram(name, parameter.detach().flatten(), step)
                    if parameter.grad is not None:
                        self.writer.add_histogram(
                            name + "_gradient",
                            parameter.grad.detach().flatten(),
                            step,
                        )

            self.log.info(
                f"{self.step_str} record monitor time: {time.perf_counter()-rec_monitor_start:10.3e}s"
            )

    # check if stopping criterion is met
    def _check_stopping_criterion(self, loss, losses, step):
        if self.manager.rank == 0:
            if self.stop_criterion_metric is None:
                return False
            elif step % self.stop_criterion_freq == 0:
                criterion_metric_dict = {
                    "loss": {"loss": float(loss.cpu().detach().numpy())}
                }
                criterion_metric_dict["loss"].update(
                    {key: float(val.cpu().detach()) for key, val in losses.items()}
                )
                if self.has_monitors:
                    criterion_metric_dict.update(
                        {
                            "monitor": {
                                key: float(val.cpu().detach().numpy())
                                for key, val in self.monitor_outvar.items()
                            }
                        }
                    )
                if self.has_validators:
                    criterion_metric_dict.update(
                        {
                            "validation": {
                                key: float(val.cpu().detach().numpy())
                                for key, val in self.validator_outvar.items()
                            }
                        }
                    )
                stop_training = self.stop_criterion.evaluate(criterion_metric_dict)
                return stop_training
            else:
                return False

    def _train_loop(
        self,
        sigterm_handler=None,
    ):  # TODO this train loop may be broken up into methods if need for future children classes

        # make directory if doesn't exist
        if self.manager.rank == 0:
            # exist_ok=True to skip creating directory that already exists
            os.makedirs(self.network_dir, exist_ok=True)

        # create global model for restoring and saving
        self.saveable_models = self.get_saveable_models()
        self.global_optimizer_model = self.create_global_optimizer_model()

        # initialize optimizer from hydra
        self.compute_gradients = getattr(
            self, self.cfg.optimizer._params_.compute_gradients
        )
        self.apply_gradients = getattr(
            self, self.cfg.optimizer._params_.apply_gradients
        )

        # initialize optimizer and scheduler from hydra
        self.optimizer, self.scheduler = instantiate_optim(
            self.cfg, model=self.global_optimizer_model
        )

        # initialize aggregator from hydra
        self.aggregator = instantiate_agg(
            self.cfg,
            model=self.global_optimizer_model.parameters(),
            num_losses=self.get_num_losses(),
        )

        if len(list(self.aggregator.parameters())) > 0:
            self.log.debug("Adding loss aggregator param group. LBFGS will not work!")
            self.optimizer.add_param_group(
                {"params": list(self.aggregator.parameters())}
            )

        # create grad scalar for AMP
        # grad scaler is only available for float16 dtype on cuda device
        enable_scaler = self.amp and self.amp_dtype == "float16"
        self.scaler = GradScaler(
            enable=enable_scaler, incr_every_n_steps=2000, init_loss_scaling=2**16
        )
        self.auto_cast_ctx = (
            paddle.amp.auto_cast(enable=self.amp, dtype=self.amp_dtype)
            if self.amp
            else nullcontext()
        )

        self.enable_scaler = enable_scaler
        # make stop criterion
        if self.stop_criterion_metric is not None:
            self.stop_criterion = StopCriterion(
                self.stop_criterion_metric,
                self.stop_criterion_min_delta,
                self.stop_criterion_patience,
                self.stop_criterion_mode,
                self.stop_criterion_freq,
                self.stop_criterion_strict,
                self.cfg.training.rec_monitor_freq,
                self.cfg.training.rec_validation_freq,
            )

        # load network
        self.initial_step = self.load_network()

        # make summary writer
        self.writer = SummaryWriter(
            log_dir=self.network_dir, purge_step=self.summary_freq + 1
        )
        self.summary_histograms = self.cfg["summary_histograms"]

        # write tensorboard config
        if self.manager.rank == 0:
            self.writer.add_text(
                "config", f"<pre>{str(OmegaConf.to_yaml(self.cfg))}</pre>"
            )

        # create profiler
        try:
            self.profile = self.cfg.profiler.profile
            self.profiler_start_step = self.cfg.profiler.start_step
            self.profiler_end_step = self.cfg.profiler.end_step
            if self.profiler_end_step < self.profiler_start_step:
                self.profile = False
        except:
            self.profile = False
            self.profiler_start_step = -1
            self.profiler_end_step = -1

        # Distributed barrier before starting the train loop
        if self.manager.distributed:
            dist.barrier()
        barrier_flag = False

        if self.manager.cuda:
            start_event = paddle.device.cuda.Event(enable_timing=True)
            end_event = paddle.device.cuda.Event(enable_timing=True)
            start_event.record()
            t = time.perf_counter()
        else:
            t = time.perf_counter()

        # termination signal handler
        if sigterm_handler is None:
            self.sigterm_handler = lambda: False
        else:
            self.sigterm_handler = sigterm_handler

        # train loop
        with ExitStack() as stack:
            if self.profile:
                raise NotImplementedError(
                    "Profiler is not supported for Modulus with Paddle backend."
                )
                # Add NVTX context if in profile mode

            for step in range(self.initial_step, self.max_steps + 1):

                if self.sigterm_handler():
                    if self.manager.rank == 0:
                        self.log.info(
                            f"Training terminated by the user at iteration {step}"
                        )
                    break

                if self.profile and step == self.profiler_start_step:
                    # Start profiling
                    self.log.info("Starting profiler at step {}".format(step))
                    paddle.framework.core.nvprof_start()
                    paddle.framework.core.nvprof_enable_record_event()

                if self.profile and step == self.profiler_end_step:
                    # Stop profiling
                    self.log.info("Stopping profiler at step {}".format(step))
                    paddle.framework.core.nvprof_stop()

                paddle.framework.core.nvprof_nvtx_push("Training iteration")

                if self.cfg.cuda_graphs:
                    raise NotImplementedError(
                        "CUDA-graphs are not supported for Modulus with Paddle backend."
                    )
                    # NOTE: CUDA-graph is not supported yet
                    # If cuda graphs statically load it into defined allocations
                    self.load_data(static=True)

                    loss, losses = self._cuda_graph_training_step(step)
                else:
                    # Load all data for constraints
                    self.load_data()

                    self.optimizer.clear_grad()

                    # compute gradients
                    loss, losses = self.compute_gradients(
                        self.aggregator, self.global_optimizer_model, step
                    )

                    # take optimizer step
                    self.apply_gradients()

                    # take scheduler step
                    if hasattr(self.scheduler, "step"):
                        self.scheduler.step()

                # check for nans in loss
                if paddle.isnan(loss):
                    self.log.error("loss went to Nans")
                    break

                self.step_str = f"[step: {step:10d}]"

                # write train loss / learning rate tensorboard summaries
                if step % self.summary_freq == 0:
                    if self.manager.rank == 0:

                        # add train loss scalars
                        for key, value in losses.items():
                            if TF_SUMMARY:
                                self.writer.add_scalar(
                                    "Train_/loss_L2" + str(key),
                                    float(value),
                                    step,
                                )
                            else:
                                self.writer.add_scalar(
                                    "Train/loss_" + str(key),
                                    float(value),
                                    step,
                                )
                        if TF_SUMMARY:
                            self.writer.add_scalar("Optimzer/loss", loss, step)
                            self.writer.add_scalar(
                                "learning_rate/lr",
                                self.optimizer.get_lr(),
                                step,
                            )
                        else:
                            self.writer.add_scalar(
                                "Train/loss_aggregated", float(loss), step
                            )
                            self.writer.add_scalar(
                                "Train/learning_rate",
                                self.optimizer.get_lr(),  # TODO: handle list
                                step,
                            )

                    if self.manager.distributed:
                        barrier_flag = True

                # write train / inference / validation datasets to tensorboard and file
                if step % self.cfg.training.rec_constraint_freq == 0:
                    barrier_flag = True
                    self._record_constraints()

                if (step % self.cfg.training.rec_validation_freq == 0) and (
                    self.has_validators
                ):
                    barrier_flag = True
                    self._record_validators(step)

                if (step % self.cfg.training.rec_inference_freq == 0) and (
                    self.has_inferencers
                ):
                    barrier_flag = True
                    self._record_inferencers(step)

                if (step % self.cfg.training.rec_monitor_freq == 0) and (
                    self.has_monitors
                ):
                    barrier_flag = True
                    self._record_monitors(step)

                # save checkpoint
                if step % self.save_network_freq == 0:
                    # Get data parallel rank so all processes in the first model parallel group
                    # can save their checkpoint. In the case without model parallelism, data_parallel_rank
                    # should be the same as the process rank itself
                    data_parallel_rank = (
                        self.manager.group_rank("data_parallel")
                        if self.manager.distributed
                        else 0
                    )
                    if data_parallel_rank == 0:
                        self.save_checkpoint(step)
                        self.log.info(
                            f"{self.step_str} saved checkpoint to {add_hydra_run_path(self.network_dir)}"
                        )

                    if self.manager.distributed:
                        barrier_flag = True

                if self.manager.distributed and barrier_flag:
                    dist.barrier()
                    barrier_flag = False

                # print loss stats
                if step % self.print_stats_freq == 0:
                    # synchronize and get end time
                    if self.manager.cuda:
                        end_event.record()
                        end_event.synchronize()
                        # elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
                        t_end = time.perf_counter()
                        elapsed_time = (t_end - t) * 1000.0  # in milliseconds
                        t = time.perf_counter()
                    else:
                        t_end = time.perf_counter()
                        elapsed_time = (t_end - t) * 1000.0  # in milliseconds
                        t = time.perf_counter()
                    # Reduce loss across all GPUs
                    if self.manager.distributed:
                        dist.reduce(loss, 0, op=dist.ReduceOp.AVG)
                        elapsed_time = paddle.to_tensor(elapsed_time, place=self.place)
                        dist.reduce(elapsed_time, 0, op=dist.ReduceOp.AVG)
                        elapsed_time = float(elapsed_time)

                    # print statement
                    print_statement = f"{self.step_str} lr: {self.optimizer.get_lr():10.3e}, loss: {float(loss):10.3e}"
                    eta_sec = (
                        (self.max_steps - step)
                        * (elapsed_time / self.print_stats_freq)
                        / 1000
                    )
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                    if step >= self.initial_step + self.print_stats_freq:
                        print_statement += f", time/iteration: {elapsed_time / self.print_stats_freq:10.3e} ms, ETA: {eta_str}"
                    if self.manager.rank == 0:
                        self.log.info(print_statement)

                    if self.manager.cuda:
                        start_event.record()
                    else:
                        t = time.perf_counter()

                # check stopping criterion
                stop_training = self._check_stopping_criterion(loss, losses, step)
                if stop_training:
                    if self.manager.rank == 0:
                        self.log.info(
                            f"{self.step_str} stopping criterion is met, finished training!"
                        )
                    break

                # check max steps
                if step >= self.max_steps:
                    if self.manager.rank == 0:
                        self.log.info(
                            f"{self.step_str} reached maximum training steps, finished training!"
                        )
                    break

                paddle.framework.core.nvprof_nvtx_pop()

    def _cuda_graph_training_step(self, step: int):
        raise NotImplementedError("CUDA graph training is not implemented yet")

    def _eval(
        self,
    ):

        # check the directory exists
        if not os.path.exists(self.network_dir):
            raise RuntimeError("Network checkpoint is required for eval mode.")

        # create global model for restoring and saving
        self.saveable_models = self.get_saveable_models()

        # set device
        if self.place is None:
            self.place = self.manager.place

        # load model
        self.step = self.load_step()
        self.step = self.load_model()
        self.step_str = f"[step: {self.step:10d}]"

        # make summary writer
        self.writer = SummaryWriter(
            log_dir=self.network_dir, purge_step=self.summary_freq + 1
        )
        self.summary_histograms = self.cfg["summary_histograms"]

        if self.manager.cuda:
            paddle.device.synchronize()

        # write inference / validation datasets to tensorboard and file
        if self.has_validators:
            self._record_validators(self.step)
        if self.has_inferencers:
            self._record_inferencers(self.step)
        if self.has_monitors:
            self._record_monitors(self.step)

    def _stream(
        self,
    ):

        # check the directory exists
        if not os.path.exists(self.network_dir):
            raise RuntimeError("Network checkpoint is required for stream mode.")

        # create global model for restoring and saving
        self.saveable_models = self.get_saveable_models()

        # set device
        if self.place is None:
            self.place = self.manager.place

        # load model
        self.step = self.load_step()
        self.step = self.load_model()
        self.step_str = f"[step: {self.step:10d}]"

        if self.manager.cuda:
            paddle.device.synchronize()

        # write streamed results to file
        return self.record_stream

    @staticmethod
    def _load_network(
        initialization_network_dir: str,
        network_dir: str,
        models: List[nn.Layer],
        optimizer: Optimizer,
        aggregator: nn.Layer,
        scheduler: LRScheduler,
        scaler: GradScaler,
        log: logging.Logger,
        manager: DistributedManager,
        device: Optional[str] = None,
    ):
        # set device
        if device is None:
            device = manager.place

        # load optimizer
        step = Trainer._load_optimizer(
            network_dir,
            optimizer,
            aggregator,
            scheduler,
            scaler,
            log,
            device,
        )

        # load model
        step = Trainer._load_model(
            initialization_network_dir,
            network_dir,
            models,
            step,
            log,
            device,
        )
        return step

    @staticmethod
    def _load_optimizer(
        network_dir: str,
        optimizer: Optimizer,
        aggregator: nn.Layer,
        scheduler: LRScheduler,
        scaler: GradScaler,
        log: logging.Logger,
        device: str,
    ):
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        # attempt to restore optimizer
        optimizer_checkpoint_file = (
            network_dir + f"/optim_checkpoint.{model_parallel_rank}.pdparams"
        )
        log.info("attempting to restore from: " + add_hydra_run_path(network_dir))
        if os.path.exists(optimizer_checkpoint_file):
            try:
                checkpoint = paddle.load(optimizer_checkpoint_file)
                optimizer.set_state_dict(checkpoint["optimizer_state_dict"])
                aggregator.set_state_dict(checkpoint["aggregator_state_dict"])
                scheduler.set_state_dict(checkpoint["scheduler_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                step = checkpoint["step"]
                success = colored("Success loading optimizer: ", "green")
                log.info(success + add_hydra_run_path(optimizer_checkpoint_file))
            except:
                fail = colored("Fail loading optimizer: ", "red")
                step = 0
                log.info(
                    fail
                    + add_hydra_run_path(network_dir + "/optim_checkpoint.pdparams")
                )
        else:
            log.warning("optimizer checkpoint not found")
            step = 0
        return step

    @staticmethod
    def _load_model(
        initialization_network_dir: str,
        network_dir: str,
        models: List[nn.Layer],
        step: int,
        log: logging.Logger,
        device: str,
    ):
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        # attempt to restrore from initialization network dir
        if initialization_network_dir != "":
            for i_dir in initialization_network_dir.split(","):
                if os.path.exists(i_dir):
                    log.info("attempting to initialize network from " + i_dir)
                    for model in models:
                        if os.path.exists(i_dir + "/" + model.checkpoint_filename):
                            try:
                                model.load(i_dir, map_location=device)
                                success = colored("Success loading model: ", "green")
                                log.info(
                                    success + i_dir + "/" + model.checkpoint_filename
                                )
                            except:
                                fail = colored("Fail loading model: ", "red")
                                step = 0
                                log.error(
                                    fail + i_dir + "/" + model.checkpoint_filename
                                )
                        else:
                            log.warning(
                                "model "
                                + model.checkpoint_filename
                                + " not found for initialization"
                            )

        # attempt to restore models
        for model in models:
            if os.path.exists(network_dir + "/" + model.checkpoint_filename):
                try:
                    model.load(network_dir, map_location=device)
                    success = colored("Success loading model: ", "green")
                    log.info(
                        success
                        + add_hydra_run_path(
                            network_dir + "/" + model.checkpoint_filename
                        )
                    )
                except:
                    fail = colored("Fail loading model: ", "red")
                    log.info(
                        fail
                        + add_hydra_run_path(
                            network_dir + "/" + model.checkpoint_filename
                        )
                    )
            else:
                log.warning("model " + model.checkpoint_filename + " not found")
                step = 0
        return step

    @staticmethod
    def _load_step(
        network_dir: str,
        device: Optional[str] = None,
    ):
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        if os.path.exists(
            network_dir + f"/optim_checkpoint.{model_parallel_rank}.pdparams"
        ):
            try:
                checkpoint = paddle.load(
                    os.path.join(
                        network_dir, f"optim_checkpoint.{model_parallel_rank}.pdparams"
                    )
                )
                step = checkpoint["step"]
            except:
                step = 0
        else:
            step = 0
        return step

    @staticmethod
    def _save_checkpoint(
        network_dir: str,
        models: List[nn.Layer],
        optimizer: Optimizer,
        aggregator: nn.Layer,
        scheduler: LRScheduler,
        scaler: GradScaler,
        step: int,
    ):
        # Get model parallel rank so all processes in the first model parallel group
        # can save their checkpoint. In the case without model parallelism, model_parallel_rank
        # should be the same as the process rank itself and only rank 0 saves
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        # save models
        for model in models:
            # model.save(network_dir, step)
            model.save(network_dir, step)

        # save step, optimizer, aggregator, and scaler
        paddle.save(
            {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "aggregator_state_dict": aggregator.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            os.path.join(
                network_dir, f"{step}_optim_checkpoint.{model_parallel_rank}.pdparams"
            ),
        )
