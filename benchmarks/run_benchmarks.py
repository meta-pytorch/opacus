# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import argparse
import itertools
import json
import logging
from os.path import exists
from typing import Any, Dict

from benchmarks.benchmark_layer import run_layer_benchmark
from benchmarks.layers import LayerType
from benchmarks.utils import get_layer_set, get_path, save_results


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_and_save_benchmark(
    layer: LayerType,
    batch_size: int,
    gsm_mode: str,
    args,
    layer_config: Dict[str, Any],
    root: str = "./results/raw/",
    suffix: str = "",
) -> None:
    """Runs and saves (if desired) the benchmark for the given layer and batch size.

    Args:
        layer: the layer to run benchmarks for
        batch_size: the batch size to run benchmarks for
        args: additional arguments
        layer_config: the settings for this layer
        root: directory to write results to
        suffix: optional string to append to result file name
    """

    logger.info(f"Benchmarking {layer} {gsm_mode} layer with batch size {batch_size}.")
    results = []

    for i in range(args.num_runs):
        runtime, memory_stats = run_layer_benchmark(
            num_repeats=args.num_repeats,
            forward_only=args.forward_only,
            layer_name=layer,
            batch_size=batch_size,
            random_seed=args.random_seed + i if args.random_seed is not None else None,
            gsm_mode=gsm_mode,
            **layer_config,
        )
        res = {"runtime": runtime, "memory_stats": memory_stats}
        results.append(res)
        logger.info(res)

    # save the benchmark results if desired
    if not args.no_save:
        save_results(
            layer=layer,
            batch_size=batch_size,
            num_runs=args.num_runs,
            num_repeats=args.num_repeats,
            gsm_mode=gsm_mode,
            results=results,
            config=layer_config,
            random_seed=args.random_seed,
            forward_only=args.forward_only,
            root=root,
            suffix=suffix,
        )


def main(args) -> None:
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # get config for each layer
    logger.info(f"Using {args.config_file} as config file.")
    with open(args.config_file) as config_file:
        config = json.load(config_file)

    for layer, batch_size, gsm_mode in itertools.product(
        args.layers, args.batch_sizes, args.grad_sample_modes
    ):
        # skip benchmark for this layer and batch size if applicable
        if args.cont and exists(
            get_path(
                layer=layer,
                batch_size=batch_size,
                num_runs=args.num_runs,
                num_repeats=args.num_repeats,
                random_seed=args.random_seed,
                forward_only=args.forward_only,
                root=args.root,
                suffix=args.suffix,
                gsm_mode=gsm_mode,
            )
        ):
            logger.info(
                f"Skipping {layer} ({gsm_mode}) at {batch_size} - already exists."
            )
            continue

        try:
            # run and save (if applicable) the benchmark for this layer and batch size
            run_and_save_benchmark(
                layer=layer,
                batch_size=batch_size,
                args=args,
                layer_config=config[get_layer_set(layer)],
                root=args.root,
                suffix=args.suffix,
                gsm_mode=gsm_mode,
            )
        except Exception as e:
            logger.info(
                f"Skipping {layer} ({gsm_mode}) at {batch_size} - Failed with {e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    layers = [v for k, v in LayerType.__dict__.items() if not k.startswith("__")]
    parser.add_argument("--layers", nargs="+", default=layers, type=str, choices=layers)
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        type=int,
    )
    parser.add_argument(
        "--num_runs",
        default=100,
        type=int,
        help="number of benchmarking runs for each layer and batch size",
    )
    parser.add_argument(
        "--num_repeats",
        default=20,
        type=int,
        help="number of forward/backward passes per run",
    )
    parser.add_argument(
        "--forward_only", action="store_true", help="only run forward passes"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="random seed for the first run for each layer and batch size, "
        "subsequent runs increase the random seed by 1",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        default="config.json",
        type=str,
        help="path to config file with settings for each layer",
    )
    parser.add_argument(
        "--cont", action="store_true", help="only run missing experiments"
    )
    parser.add_argument(
        "--root",
        default="./results/raw/",
        type=str,
        help="path to directory where benchmark results should be saved",
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="suffix to append to each result file's name",
    )
    parser.add_argument(
        "--grad_sample_modes",
        type=str,
        nargs="+",
        choices=["baseline", "hooks", "ew", "functorch"],
        default=["baseline", "hooks"],
        help="Mode to compute per sample gradinets: "
        "Classic (hooks), Functorch(functorch), "
        "ExpandedWeights(ew), Non-private(baseline)",
    )
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
