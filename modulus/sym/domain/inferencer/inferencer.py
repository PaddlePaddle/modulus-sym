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


class Inferencer:
    """
    Inferencer base class
    """

    def forward_grad(self, invar):
        pred_outvar = self.model(invar)
        return pred_outvar

    def forward_nograd(self, invar):
        with paddle.no_grad():
            pred_outvar = self.model(invar)
            return pred_outvar

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        raise NotImplementedError("Subclass of Inferencer needs to implement this")
