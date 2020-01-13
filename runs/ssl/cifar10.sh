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

echo
echo "# Standard"
for seed in 1 2 3 4 5; do for size in 250 1000 4000; do
  echo "python cta/cta_remixmatch.py --dataset=cifar10.${seed}@{$size}-1 --K=8"
done; done

for seed in 1 2 3 4 5; do
  echo "python cta/cta_remixmatch.py --dataset=cifar10.${seed}@40-1 --K=1 --w_rot=2 --warmup_kimg=8192"
done
