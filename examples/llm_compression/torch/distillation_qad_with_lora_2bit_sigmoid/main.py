# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
from pathlib import Path
from typing import Any

from nncf.parameters import CompressionFormat
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.quantization.layers import SymmetricLoraSigmoidQuantizer


def _load_base_example() -> Any:
    base_path = Path(__file__).parents[1] / "distillation_qad_with_lora_2_3_bit" / "main.py"
    module_spec = importlib.util.spec_from_file_location("distillation_qad_with_lora_base", base_path)
    if module_spec is None or module_spec.loader is None:
        msg = f"Unable to load base distillation example at {base_path}."
        raise ImportError(msg)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


base = _load_base_example()
_original_check_args = base.check_args
_original_compress_weights = base.compress_weights
_original_run_training = base.run_training
_temperature_config = {"initial": 0.1, "final": 0.0005}


def get_argument_parser():
    """
    Returns the base distillation parser extended with sigmoid temperature controls.

    :return: Argument parser for two-bit sigmoid LoRA training.
    """
    parser = _original_get_argument_parser()
    parser.set_defaults(bits=2)
    parser.add_argument("--initial_temperature", type=float, default=0.1, help="Initial sigmoid temperature.")
    parser.add_argument("--final_temperature", type=float, default=0.0001, help="Final sigmoid temperature.")
    return parser


_original_get_argument_parser = base.get_argument_parser


def check_args(args) -> None:
    """
    Validates two-bit sigmoid LoRA arguments and records its temperature schedule.

    :param args: Parsed command-line arguments.
    """
    _original_check_args(args)
    if args.bits != 2:
        msg = "The sigmoid LoRA example supports only --bits 2."
        raise ValueError(msg)
    if args.initial_temperature <= 0 or args.final_temperature <= 0:
        msg = "Sigmoid temperatures must be positive."
        raise ValueError(msg)
    _temperature_config["initial"] = args.initial_temperature
    _temperature_config["final"] = args.final_temperature


def compress_weights_sigmoid(*args, **kwargs):
    """
    Compresses weights using the two-bit sigmoid LoRA format.

    :param args: Positional arguments forwarded to NNCF compression.
    :param kwargs: Keyword arguments forwarded to NNCF compression.
    :return: Compressed model.
    """
    kwargs["compression_format"] = CompressionFormat.FQ_LORA_SIGMOID
    return _original_compress_weights(*args, **kwargs)


def _set_temperature(model, temperature: float) -> None:
    for _, quantizer in get_hook_storage(model).named_hooks():
        if isinstance(quantizer, SymmetricLoraSigmoidQuantizer):
            quantizer.set_temperature(temperature)


def _get_temperature(backpropagation_step: int, total_backpropagation_steps: int) -> float:
    """
    Calculates the linearly scheduled sigmoid temperature.

    :param backpropagation_step: Zero-based index of the upcoming backpropagation step.
    :param total_backpropagation_steps: Total number of backpropagation steps in the training phase.
    :return: The temperature for the specified step.
    """
    initial_temperature = _temperature_config["initial"]
    final_temperature = _temperature_config["final"]
    if total_backpropagation_steps <= 1:
        return initial_temperature
    if backpropagation_step >= total_backpropagation_steps - 1:
        return final_temperature
    progress = backpropagation_step / (total_backpropagation_steps - 1)
    return initial_temperature + (final_temperature - initial_temperature) * progress


def run_training_with_temperature(*args, **kwargs):
    """
    Runs the base training loop with linearly scheduled sigmoid temperature annealing.

    :param args: Positional arguments forwarded to the base training loop.
    :param kwargs: Keyword arguments forwarded to the base training loop.
    :return: Total optimizer steps completed.
    """
    total_backpropagation_steps = kwargs["num_epochs"] * kwargs["microbatches_per_epoch"]
    backpropagation_step = 0

    def set_temperature_before_forward(_module: Any, _inputs: tuple[Any, ...]) -> None:
        nonlocal backpropagation_step
        del _module, _inputs
        temperature = _get_temperature(backpropagation_step, total_backpropagation_steps)
        _set_temperature(kwargs["model"], temperature)
        backpropagation_step += 1

    temperature_hook = kwargs["model"].register_forward_pre_hook(set_temperature_before_forward)
    try:
        return _original_run_training(*args, **kwargs)
    finally:
        temperature_hook.remove()


base.get_argument_parser = get_argument_parser
base.check_args = check_args
base.compress_weights = compress_weights_sigmoid
base.run_training = run_training_with_temperature


if __name__ == "__main__":
    base.main(sys.argv[1:])
