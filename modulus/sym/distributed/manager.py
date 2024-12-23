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

import paddle
import paddle.distributed as dist

import logging
import os
import time
import numpy as np

logger = logging.getLogger("__name__")


# Create singleton DistributedManager class
class DistributedManager(object):
    _shared_state = {}

    def __new__(cls):
        obj = super(DistributedManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_rank"):
            obj._rank = 0
        if not hasattr(obj, "_world_size"):
            obj._world_size = 1
        if not hasattr(obj, "_local_rank"):
            obj._local_rank = 0
        if not hasattr(obj, "_distributed"):
            obj._distributed = False
        if not hasattr(obj, "_device"):
            obj._device: str = str(
                f"gpu:0" if paddle.device.cuda.device_count() >= 1 else "cpu"
            )
        if not hasattr(obj, "_cuda"):
            obj._cuda = paddle.device.cuda.device_count() >= 1
        if not hasattr(obj, "_broadcast_buffers"):
            obj._broadcast_buffers = False
        if not hasattr(obj, "_find_unused_parameters"):
            obj._find_unused_parameters = False
        if not hasattr(obj, "_cuda_graphs"):
            obj._cuda_graphs = False
        obj.place = obj._device
        return obj

    @property
    def rank(self):
        return self._rank

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    @property
    def distributed(self):
        return self._distributed

    @property
    def cuda(self):
        return self._cuda

    @property
    def group_names(self):
        """
        Returns a list of all named process groups created
        """
        return self._groups.keys()

    def group(self, name=None):
        """
        Returns a process group with the given name
        If name is None, group is also None indicating the default process group
        If named group does not exist, returns None also
        """
        if name in self._groups.keys():
            return self._groups[name]
        else:
            return None

    def group_size(self, name=None):
        """
        Returns the size of named process group
        """
        if name is None:
            return self._world_size
        group = self.group(name)
        return dist.get_world_size(group=group)

    def group_rank(self, name=None):
        """
        Returns the rank in named process group
        """
        if name is None:
            return self._rank
        group = self.group(name)
        return dist.get_rank(group=group)

    def group_name(self, group=None):
        """
        Returns the name of process group
        """
        if group is None:
            return None
        return self._group_names[group]

    @property
    def broadcast_buffers(self):
        return self._broadcast_buffers

    @broadcast_buffers.setter
    def broadcast_buffers(self, broadcast: bool):
        self._broadcast_buffers = broadcast

    @property
    def find_unused_parameters(self):
        return self._find_unused_parameters

    @find_unused_parameters.setter
    def find_unused_parameters(self, find_params: bool):
        if find_params:
            # Logger may not be config'd here yet
            logger.warning(
                "Setting `find_unused_parameters` in DDP to true, use only if necessary."
            )
        self._find_unused_parameters = find_params

    @property
    def cuda_graphs(self):
        return self._cuda_graphs

    @cuda_graphs.setter
    def cuda_graphs(self, graphs: bool):
        # Function for any modifications needed for DDP using cuda graphs
        if graphs and self._find_unused_parameters:
            # Logger may not be config'd here yet
            logger.warning(
                "DDP `find_unused_parameters` must be false for CUDA graphs."
            )
            raise ValueError(
                "`cuda_graphs` and `find_unused_parameters` cannot both be true"
            )

        self._cuda_graphs = graphs

    @staticmethod
    def get_available_backend():
        if paddle.device.cuda.device_count() >= 1 and (
            not dist.parallel.is_initialized() or dist.get_backend() == "NCCL"
        ):
            return "nccl"
        else:
            return "gloo"

    @staticmethod
    def initialize_env():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size > 1:
            dist.init_parallel_env()
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ.get("LOCAL_RANK"))
        else:
            local_rank = rank % paddle.device.cuda.device_count()
        addr = None
        port = None

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
        )

    @staticmethod
    def initialize_open_mpi(addr, port):
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="openmpi",
        )

    @staticmethod
    def initialize_slurm(port):
        rank = int(os.environ.get("SLURM_PROCID"))
        world_size = int(os.environ.get("SLURM_NPROCS"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="slurm",
        )

    @staticmethod
    def initialize():
        addr = os.getenv("MASTER_ADDR", "localhost")
        port = os.getenv("MASTER_PORT", "12355")
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        try:
            DistributedManager.initialize_env()
        except Exception as e:
            raise e
            if "SLURM_PROCID" in os.environ:
                DistributedManager.initialize_slurm(port)
            elif "OMPI_COMM_WORLD_RANK" in os.environ:
                DistributedManager.initialize_open_mpi(addr, port)

        # Set per rank numpy random seed for data sampling
        np.random.seed(seed=DistributedManager().rank)

        manager = DistributedManager()
        if manager.distributed:
            print(
                f'Initialized process {manager.rank} of {manager.world_size} using method "{manager._initialization_method}". Device set to {str(manager.device)}'
            )

    @staticmethod
    def setup(
        rank=0,
        world_size=1,
        local_rank=None,
        addr="localhost",
        port="12355",
        backend="nccl",
        method="env",
    ):
        # os.environ["MASTER_ADDR"] = addr
        # os.environ["MASTER_PORT"] = str(port)

        manager = DistributedManager()

        manager._distributed = (world_size > 1) and dist.is_available()
        if manager._distributed:
            # Update rank and world_size if using distributed
            manager._rank = rank
            manager._world_size = world_size
            if local_rank is None:
                manager._local_rank = rank % paddle.device.cuda.device_count()
            else:
                manager._local_rank = local_rank

            # Setup distributed process group
            # time.sleep(1)
            # dist.init_parallel_env()

        manager._groups = {}
        manager._group_ranks = {}
        manager._group_names = {}

        manager._device = str(
            f"gpu:{manager.local_rank}"
            if paddle.device.cuda.device_count() >= 1
            else "cpu"
        )
        # Needed for cuda graphs
        if paddle.device.cuda.device_count() >= 1:
            paddle.device.set_device(device=f"gpu:{manager.local_rank}")

        manager._initialization_method = method

        # Set device for this process and empty cache to optimize memory usage
        paddle.device.set_device(manager.place)
        paddle.device.cuda.empty_cache()

    @staticmethod
    def create_process_subgroup(name: str, size: int, group_name=None, verbose=False):

        manager = DistributedManager()
        if not manager.distributed:
            return None

        assert name not in manager._groups, f"Group with name {name} already exists"

        # Get parent group's params
        group = manager._group[group_name] if group_name else None
        group_size = dist.get_world_size(group=group)
        group_rank = dist.get_rank(group=group)
        num_groups = manager.world_size // group_size

        # Get number of sub-groups per parent group
        assert (
            group_size % size == 0
        ), f"Cannot divide group size {group_size} evenly into subgroups of size {size}"
        num_subgroups = group_size // size

        # Create all the sub-groups
        # Note: all ranks in the job need to create all sub-groups in
        # the same order even if a rank is not part of a sub-group
        manager._group_ranks[name] = []
        for g in range(num_groups):
            for i in range(num_subgroups):
                # Get global ranks that are part of this sub-group
                start = i * size
                end = start + size
                if group_name:
                    ranks = manager._group_ranks[group_name][g][start:end]
                else:
                    ranks = list(range(start, end))
                # Create sub-group and keep track of ranks
                tmp_group = dist.new_group(ranks=ranks)
                manager._group_ranks[name].append(ranks)
                if manager.rank in ranks:
                    # Set group in manager only if this rank is part of the group
                    manager._groups[name] = tmp_group
                    manager._group_names[tmp_group] = name

        if verbose and manager.rank == 0:
            print(f"Process group '{name}':")
            for grp in manager._group_ranks[name]:
                print("    ", grp)

    @staticmethod
    def create_orthogonal_process_group(name: str, group_name: str, verbose=False):
        manager = DistributedManager()
        if not manager.distributed:
            return None

        assert (
            group_name in manager._groups
        ), f"Group with name {group_name} does not exist"
        assert name not in manager._groups, f"Group with name {name} already exists"

        group_ranks = manager._group_ranks[group_name]
        orthogonal_ranks = [list(i) for i in zip(*group_ranks)]

        for ranks in orthogonal_ranks:
            tmp_group = dist.new_group(ranks=ranks)
            if manager.rank in ranks:
                # Set group in manager only if this rank is part of the group
                manager._groups[name] = tmp_group
                manager._group_names[tmp_group] = name

        manager._group_ranks[name] = orthogonal_ranks

        if verbose and manager.rank == 0:
            print(f"Process group '{name}':")
            for grp in manager._group_ranks[name]:
                print("    ", grp)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()
