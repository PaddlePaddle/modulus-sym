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

import functools
import hydra
import os
import paddle
import logging
import copy
import pprint
import numpy as np
import random

from termcolor import colored
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, MISSING
from typing import Optional, Any, Union, List, Tuple
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from modulus.sym.key import Key
from modulus.sym.models.arch import Arch
from modulus.sym.distributed import DistributedManager
from modulus.sym.models.utils import ModulusModels
from modulus.sym.models.activation import Activation

from .arch import ModelConf
from .config import register_modulus_configs, ModulusConfig
from .hydra import register_hydra_configs
from .loss import register_loss_configs
from .metric import register_metric_configs
from .arch import register_arch_configs
from .optimizer import register_optimizer_configs
from .pde import register_pde_configs
from .profiler import register_profiler_configs
from .scheduler import register_scheduler_configs
from .training import register_training_configs
from .callbacks import register_callbacks_configs
from .graph import register_graph_configs


logger = logging.getLogger(__name__)


def main(config_path: str, config_name: str = "config"):
    """Modified decorator for loading hydra configs in modulus
    See: https://github.com/facebookresearch/hydra/blob/main/hydra/main.py
    """

    def register_decorator(func):
        @functools.wraps(func)
        def func_decorated(cfg_passthrough: Optional[DictConfig] = None) -> Any:

            # # Fix all random seed
            # GLOBAL_RANDOM_SEED = 42
            # random.seed(GLOBAL_RANDOM_SEED)  # Python random module.
            # np.random.seed(GLOBAL_RANDOM_SEED)  # Numpy module.
            # paddle.seed(GLOBAL_RANDOM_SEED) # Paddle random.
            # print(f"✨ ✨ Set global random seed to {GLOBAL_RANDOM_SEED}")
            # os.environ['PYTHONHASHSEED'] = str(GLOBAL_RANDOM_SEED)

            # Enable prim mode
            # use_prim = os.getenv("FLAGS_prim_all", "False") == "True"
            # if use_prim:
            #     # print(f"✨ ✨ Prim = True, prim can be disabled by set 'FLAGS_prim_all=False'")
            #     paddle.framework.core.set_prim_eager_enabled(True)
            #     paddle.framework.core._set_prim_all_enabled(True)
            # else:
                # print(f"✨ ✨ Prim = False, prim can be disabled by set 'FLAGS_prim_all=False'")

            # Register all modulus groups before calling hydra main
            register_hydra_configs()
            register_callbacks_configs()
            register_loss_configs()
            register_metric_configs()
            register_arch_configs()
            register_optimizer_configs()
            register_pde_configs()
            register_profiler_configs()
            register_scheduler_configs()
            register_training_configs()
            register_modulus_configs()
            register_graph_configs()
            # paddle.set_num_threads(1)

            # Setup distributed process config
            DistributedManager.initialize()
            # Create model parallel process group
            model_parallel_size = os.getenv(
                "MODEL_PARALLEL_SIZE"
            )  # TODO: get this from config instead
            if model_parallel_size:
                # Create model parallel process group
                DistributedManager.create_process_subgroup(
                    "model_parallel", int(model_parallel_size), verbose=True
                )
                # Create data parallel process group for DDP allreduce
                DistributedManager.create_orthogonal_process_group(
                    "data_parallel", "model_parallel", verbose=True
                )

            # Pass through dict config
            if cfg_passthrough is not None:
                return func(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                args = args_parser.parse_args()
                # multiple times (--multirun)
                _run_hydra(
                    args=args_parser.parse_args(),
                    args_parser=args_parser,
                    task_function=func,
                    config_path=config_path,
                    config_name=config_name,
                )

        return func_decorated

    return register_decorator


def compose(
    config_name: Optional[str] = None,
    config_path: Optional[str] = None,
    overrides: List[str] = [],
    return_hydra_config: bool = False,
    job_name: Optional[str] = "app",
    caller_stack_depth: int = 2,
) -> DictConfig:
    """Internal Modulus config initializer and compose function.
    This is an alternative for initializing a Hydra config which should be used
    as a last ditch effort in cases where @modulus.main() cannot work. For more info
    see: https://hydra.cc/docs/advanced/compose_api/

    Parameters
    ----------
    config_name : str
        Modulus config name
    config_path : str
        Path to config file relative to the caller at location caller_stack_depth
    overrides : list of strings
        List of overrides
    return_hydra_config : bool
        Return the hydra options in the dict config
    job_name : string
        Name of program run instance
    caller_stack_depth : int
        Stack depth of this function call (needed for finding config relative to python).

    """
    # Clear already initialized hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    hydra.initialize(
        config_path,
        job_name,
        caller_stack_depth,
    )

    register_hydra_configs()
    register_callbacks_configs()
    register_loss_configs()
    register_metric_configs()
    register_arch_configs()
    register_optimizer_configs()
    register_pde_configs()
    register_profiler_configs()
    register_scheduler_configs()
    register_training_configs()
    register_modulus_configs()
    register_graph_configs()

    cfg = hydra.compose(
        config_name=config_name,
        overrides=overrides,
        return_hydra_config=return_hydra_config,
    )

    return cfg


def instantiate_arch(
    cfg: ModelConf,
    input_keys: Union[List[Key], None] = None,
    output_keys: Union[List[Key], None] = None,
    detach_keys: Union[List[Key], None] = None,
    verbose: bool = False,
    **kwargs,
) -> Arch:
    # Function for instantiating a modulus architecture with hydra
    assert hasattr(
        cfg, "arch_type"
    ), "Model configs are required to have an arch_type defined. \
        Improper architecture supplied, please make sure config \
        provided is a single arch config NOT the full hydra config!"

    try:
        # Convert to python dictionary
        model_cfg = OmegaConf.to_container(cfg, resolve=True)

        # Get model class beased on arch type
        modulus_models = ModulusModels()
        model_arch = modulus_models[model_cfg["arch_type"]]
        del model_cfg["arch_type"]

        # Add keys if present
        if not input_keys is None:
            model_cfg["input_keys"] = input_keys
        if not output_keys is None:
            model_cfg["output_keys"] = output_keys
        if not detach_keys is None:
            model_cfg["detach_keys"] = detach_keys

        # Add any additional kwargs
        for key, value in kwargs.items():
            model_cfg[key] = value

        # Init model from config dictionary
        model, param = model_arch.from_config(model_cfg)

        # Verbose printing
        if verbose:
            pp = pprint.PrettyPrinter(indent=4)
            logger.info(f"Initialized models with parameters: \n")
            pp.pprint(param)

    except Exception as e:
        fail = colored(f"Failed to initialize architecture.\n {model_cfg}", "red")
        raise Exception(fail) from e

    return model


def instantiate_optim(
    cfg: DictConfig, model: paddle.nn.Layer, verbose: bool = False
) -> Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler]:
    # Function for instantiating an optimizer with hydra
    # Remove custom parameters used internally in modulus
    optim_cfg = copy.deepcopy(cfg.optimizer)
    del optim_cfg._params_

    try:
        scheduler = instantiate_sched(cfg, None)
        optimizer = hydra.utils.instantiate(
            optim_cfg,
            parameters=model.parameters(),
            learning_rate=scheduler,
        )
    except Exception as e:
        fail = colored("Failed to initialize optimizer: \n", "red")
        logger.error(fail + to_yaml(optim_cfg))
        raise Exception(fail) from e

    if verbose:
        pp = pprint.PrettyPrinter(indent=4)
        logger.info(f"Initialized optimizer: \n")
        pp.pprint(optimizer)
        pp.pprint(scheduler)

    return optimizer, scheduler


def instantiate_sched(
    cfg: DictConfig, optimizer: paddle.optimizer.Optimizer = None
) -> paddle.optimizer.lr.LRScheduler:
    # Function for instantiating a scheduler with hydra
    sched_cfg = copy.deepcopy(cfg.scheduler)
    optim_cfg = copy.deepcopy(cfg.optimizer)
    # Default is no scheduler, so just make fixed LR
    if sched_cfg is MISSING:
        sched_cfg = {
            "_target_": "paddle.optimizer.lr.ConstantLR",
            "factor": 1.0,
        }
    if isinstance(optim_cfg.get("learning_rate", None), float):
        sched_cfg.update({"learning_rate": optim_cfg.learning_rate})
    # Handle custom cases
    if sched_cfg._target_ == "custom":
        if "tf.ExponentialLR" in sched_cfg._name_:
            sched_cfg = {
                "_target_": "paddle.optimizer.lr.ExponentialDecay",
                "learning_rate": optim_cfg.learning_rate,
                "gamma": sched_cfg.decay_rate ** (1.0 / sched_cfg.decay_steps),
            }
        else:
            logger.warn("Detected unsupported custom scheduler", sched_cfg)

    try:
        scheduler = hydra.utils.instantiate(sched_cfg)
    except Exception as e:
        fail = colored("Failed to initialize scheduler: \n", "red")
        logger.error(fail + to_yaml(sched_cfg))
        raise Exception(fail) from e

    return scheduler


def instantiate_agg(cfg: DictConfig, model: paddle.nn.Layer, num_losses: int = 1):
    # Function for instantiating a loss aggregator with hydra
    try:
        aggregator = hydra.utils.instantiate(
            cfg.loss,
            model,
            num_losses,
            _convert_="all",
        )
    except Exception as e:
        fail = colored("Failed to initialize loss aggregator: \n", "red")
        logger.error(fail + to_yaml(cfg.loss))
        raise Exception(fail) from e
    return aggregator


def to_yaml(cfg: DictConfig):
    """Converges dict config into a YML string"""
    return OmegaConf.to_yaml(cfg)


def add_hydra_run_path(path: Union[str, Path]) -> Path:
    """Prepends current hydra run path"""

    working_dir = Path(os.getcwd())
    # Working directory only present with @modulus.main()
    if HydraConfig.initialized():
        org_dir = Path(get_original_cwd())
        hydra_dir = working_dir.relative_to(org_dir) / Path(path)
    else:
        hydra_dir = working_dir / Path(path)

    if isinstance(path, str):
        hydra_dir = str(hydra_dir)
    return hydra_dir


def to_absolute_path(*args: Union[str, Path]):
    """Converts file path to absolute path based on run file location
    Modified from: https://github.com/facebookresearch/hydra/blob/main/hydra/utils.py
    """
    out = ()
    for path in args:
        p = Path(path)
        if not HydraConfig.initialized():
            base = Path(os.getcwd())
        else:
            ret = HydraConfig.get().runtime.cwd
            base = Path(ret)
        if p.is_absolute():
            ret = p
        else:
            ret = base / p

        if isinstance(path, str):
            out = out + (str(ret),)
        else:
            out = out + (ret,)

    if len(args) == 1:
        out = out[0]
    return out
