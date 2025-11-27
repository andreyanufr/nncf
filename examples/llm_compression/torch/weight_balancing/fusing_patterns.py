

class FusingPattern:
    def __init__(self, layer_for_fusing, scaled_layers):
        """
        Initialize a FusingPattern instance.
        Args:
            scaled_layers (list): A list of layers that are scaled.
            layer_for_fusing (str): The layer that will be used for fusing.
        """
        self.scaled_layers = scaled_layers
        self.layer_for_fusing = layer_for_fusing




llama_fusing = [
    FusingPattern("re:.*v_proj$", ["re:.*o_proj$"]),
    FusingPattern(
        "re:.*input_layernorm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    FusingPattern(
        "re:.*up_proj$",
        ["re:.*down_proj$"],
    ),
    FusingPattern(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_proj$", "re:.*up_proj$"],
    ),
]


phi_fusing = [
    FusingPattern(
        "re:.*input_layernorm$",
        ["re:.*qkv_proj$"],
    ),
    FusingPattern("re:.*qkv_proj$", ["re:.*o_proj$"]),
    FusingPattern(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_up_proj$"],
    ),
    FusingPattern(
        "re:.*gate_up_proj$",
        ["re:.*down_proj$"],
    ),
]


FUSING_PATTERN_REGISTRY: dict[str, list[FusingPattern]] = {
    "LlamaForCausalLM": llama_fusing,
    "MistralForCausalLM": llama_fusing,
    "Phi3ForCausalLM": phi_fusing,
    "Phi3VForCausalLM": phi_fusing,
    "Qwen2ForCausalLM": llama_fusing,
    "Qwen3ForCausalLM": llama_fusing,
}
