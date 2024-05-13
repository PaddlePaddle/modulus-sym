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

""" Modulus Managers
"""

import os
import logging
from typing import Dict, List, Union
from enum import Enum
import paddle
from packaging import version

# from modulus.sym.constants import JIT_PADDLE_VERSION

logger = logging.getLogger(__name__)


class JitArchMode(Enum):
    ALL = 0
    ONLY_ACTIVATION = 1


class JitManager(object):
    _shared_state = {}

    def __new__(cls):
        obj = super(JitManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_enabled"):
            obj._enabled = (
                paddle.__version__ == "0.0.0"
            ) and bool(os.getenv("to_static", "False").lower() == "true")
        if not hasattr(obj, "_arch_mode"):
            obj._arch_mode = JitArchMode.ONLY_ACTIVATION
        if not hasattr(obj, "_use_cinn"):
            obj._use_cinn = (
                paddle.__version__ == "0.0.0"
            )
        if not hasattr(obj, "_use_prim"):
            obj._use_prim = False
        if not hasattr(obj, "_autograd_nodes"):
            obj._autograd_nodes = False

        return obj

    @property
    def arch_mode(self):
        return self._arch_mode

    @arch_mode.setter
    def arch_mode(self, mode: str):
        if mode == "all":
            self._arch_mode = JitArchMode.ALL
        elif mode == "only_activation":
            self._arch_mode = JitArchMode.ONLY_ACTIVATION
        else:
            raise ValueError(
                f"jit arch mode should be all/only_activation, but found {mode}"
            )

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, flag):
        self._enabled = flag

    @property
    def use_cinn(self):
        return self._use_cinn

    @use_cinn.setter
    def use_cinn(self, flag):
        self._use_cinn = flag
        if flag:
            logger.info("CINN is enabled in modulus-sym(paddle backend)")
        backend = "CINN" if flag else None
        if self.enabled:
            logger.info(f"JIT using the {backend} backend")

    @property
    def use_prim(self):
        return self._use_prim

    @use_prim.setter
    def use_prim(self, flag):
        self._use_prim = flag
        if flag:
            logger.info("Prim is enabled in modulus-sym(paddle backend)")
        if self.enabled:
            if self.use_cinn and not self.use_prim:
                logger.warning(
                    f"Please set FLAGS_prim_all=True when CINN is enabled"
                )

    @property
    def autograd_nodes(self):
        return self._autograd_nodes

    @autograd_nodes.setter
    def autograd_nodes(self, flag):
        self._autograd_nodes = flag

    def __repr__(self):
        return f"JitManager: {self._shared_state}"

    def init(self, enabled, arch_mode, use_cinn, autograd_nodes):
        self.enabled = enabled
        self.arch_mode = arch_mode
        self.use_cinn = use_cinn
        self.autograd_nodes = autograd_nodes


class GraphManager(object):
    _shared_state = {}

    def __new__(cls):
        obj = super(GraphManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_func_arch"):
            obj._func_arch = False
        # TODO we should have a debug flag in the global ModulusManager
        # in the future
        if not hasattr(obj, "_debug"):
            obj._debug = False
        if not hasattr(obj, "_func_arch_allow_partial_hessian"):
            obj._func_arch_allow_partial_hessian = False

        return obj

    @property
    def func_arch(self):
        return self._func_arch

    @func_arch.setter
    def func_arch(self, flag):
        self._func_arch = flag

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, flag):
        self._debug = flag

    @property
    def func_arch_allow_partial_hessian(self):
        return self._func_arch_allow_partial_hessian

    @func_arch_allow_partial_hessian.setter
    def func_arch_allow_partial_hessian(self, flag):
        self._func_arch_allow_partial_hessian = flag

    def __repr__(self):
        return f"GraphManager: {self._shared_state}"

    def init(self, func_arch, func_arch_allow_partial_hessian, debug):
        self.func_arch = func_arch
        self.func_arch_allow_partial_hessian = func_arch_allow_partial_hessian
        self.debug = debug
