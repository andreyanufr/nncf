"""
Load an OpenVINO IR model and convert every fp32 Constant node to fp16.

A Convert node back to fp32 is inserted after each rewritten Constant so the rest
of the graph keeps its original precision contract.
"""

import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as opset


def convert_fp32_constants_to_fp16(model: ov.Model) -> int:
    """
    Replace every fp32 Constant in the given model with an fp16 Constant followed by a
    Convert back to fp32.

    :param model: OpenVINO model to modify in-place.
    :return: Number of Constants that were rewritten.
    """
    rewritten = 0
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Constant":
            continue
        if node.get_output_element_type(0) != ov.Type.f32:
            continue

        fp32_data = node.get_data()
        if fp32_data.size < 100 or len(fp32_data.shape) < 3 or fp32_data.shape[0] < 10:
            # Skip small constants since the overhead of the Convert may outweigh the memory savings.
            continue
        
        fp16_data = fp32_data.astype(np.float16)
        
        if np.abs(fp32_data - fp16_data.astype(np.float32)).max() > 0.05:
            print(f"Converting Constant '{node.get_friendly_name()}' from fp32 to fp16. Original shape: {fp32_data.shape}, original size: {fp32_data.nbytes} bytes, new size: {fp16_data.nbytes} bytes.")
            print("Difference stats: mean absolute error =", np.abs(fp32_data - fp16_data.astype(np.float32)).mean(), ", max absolute error =", np.abs(fp32_data - fp16_data.astype(np.float32)).max())
            print("Max value before conversion:", np.abs(fp32_data).max(), ", max value after conversion:", np.abs(fp16_data).max())
            print("Min value before conversion:", np.abs(fp32_data).min(), ", min value after conversion:", np.abs(fp16_data).min())

        fp16_const = opset.constant(fp16_data, dtype=ov.Type.f16, name=node.get_friendly_name())
        convert = opset.convert(fp16_const, destination_type=ov.Type.f32)
        convert.set_friendly_name(node.get_friendly_name() + "/convert_to_fp32")

        for output in node.outputs():
            for target_input in list(output.get_target_inputs()):
                target_input.replace_source_output(convert.output(0))

        rewritten += 1

    return rewritten


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to the input OpenVINO IR (.xml) file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output OpenVINO IR (.xml) file.")
    parser.add_argument(
        "--compress-to-fp16",
        action="store_true",
        help="Pass compress_to_fp16=True to ov.save_model. Off by default because constants are already fp16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    core = ov.Core()
    model = core.read_model(args.input)

    rewritten = convert_fp32_constants_to_fp16(model)
    print(f"Converted {rewritten} fp32 constants to fp16.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, args.output, compress_to_fp16=args.compress_to_fp16)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
