# Large Language Models FP8 Compression Example

This example demonstrates how to apply static FP8 quantization to [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model. It can be useful for evaluation and early HW enablement purposes.

## Prerequisites

Before running this example, ensure you have Python 3.9+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
```

## Run Example

To run example:

```bash
python main.py
```

This will automatically:

- Download the Llama-3.2-1B-Instruct model and calibration dataset
- Apply weight compression using NNCF
- Save the optimized model

```bash
lm_eval --model openvino --model_args pretrained=$file --task lambada_openai
```

| Precision  | NNCF options | acc | ppl|
| ------------- | ------------- | ------------- | ------------- |
| int8  | - | 0.6020 | 6.6077|
| int4_sym  | gs=128,SE=True,hadamard=True  | 0.5579 | 8.2816 |
| int4_sym  | gs=128,SE=True,hadamard=False  | 0.5562 | 8.9471 |
| int4_sym  | gs=128,SE=False,hadamard=False  | 0.5352 | 10.2223 |
