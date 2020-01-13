#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DEFAULT_ARGS="--dataset=cifar10.3@250-1 --train_dir experiments/Ablation --augment=d.d.d"

echo "# Vary number of augmentations"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=1"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=2"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=4"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=8"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=16"

echo "# No regularizer"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=8 --w_kl=0"
echo "# No rotation"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=8 --w_rot=0"
echo "# No Distribution matching"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --K=8 --nouse_dm"
echo "# Use L2 loss / no cross-entropy"
echo "python ablation/ab_cta_remixmatch.py $DEFAULT_ARGS --w_match=75 --K=8 --nouse_xe"

echo "# Only strong augmentations"
echo "python ablation/ab_cta_remixmatch_noweak.py $DEFAULT_ARGS --K=8"

echo "# Only weak augmentations"
echo "python ablation/ab_remixmatch.py $DEFAULT_ARGS --K=1"
echo "python ablation/ab_remixmatch.py $DEFAULT_ARGS --K=8"
echo "python ablation/ab_remixmatch.py $DEFAULT_ARGS --redux=mean --K=1"
echo "python ablation/ab_remixmatch.py $DEFAULT_ARGS --redux=mean --K=8"
