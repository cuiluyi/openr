import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from train.configs import GRPOConfig, SFTConfig


def initialize_token_embeddings_from_descriptions(
    model,
    tokenizer,
    added_tokens: list[str],
    descriptions: list[str],
) -> None:
    """
    Initialize the embeddings (input/output) and bias of newly added tokens
    using the average of their description tokens' embeddings.

    This version is compatible with DeepSpeed ZeRO-3 and safely gathers parameters.

    Args:
        model: Pretrained model.
        tokenizer: Tokenizer for converting text to token IDs.
        added_tokens: New special tokens added to tokenizer.
        descriptions: Natural language descriptions for each added token.
    """
    assert len(added_tokens) == len(descriptions), "Mismatched tokens and descriptions"

    added_token_ids = tokenizer.convert_tokens_to_ids(added_tokens)
    input_embeddings = model.get_input_embeddings()

    # Prepare output embedding layer
    output_embeddings = None
    init_output = False
    if not getattr(model.config, "tie_word_embeddings", True):
        output_embeddings = model.get_output_embeddings()
        init_output = output_embeddings is not None

    with torch.no_grad():
        for token_id, description in zip(added_token_ids, descriptions):
            # Tokenize and convert description
            desc_tokens = tokenizer.tokenize(description)
            desc_token_ids = tokenizer.convert_tokens_to_ids(desc_tokens)

            # === Input embedding init ===
            with deepspeed.zero.GatheredParameters(
                input_embeddings.weight, modifier_rank=0
            ):
                desc_input_embed = input_embeddings.weight[desc_token_ids].mean(dim=0)
                input_embeddings.weight[token_id] = desc_input_embed

            # === Output embedding init ===
            if not init_output:
                continue

            if isinstance(output_embeddings, torch.nn.Embedding):
                with deepspeed.zero.GatheredParameters(
                    output_embeddings.weight, modifier_rank=0
                ):
                    desc_output_embed = output_embeddings.weight[desc_token_ids].mean(
                        dim=0
                    )
                    output_embeddings.weight[token_id] = desc_output_embed

            elif isinstance(output_embeddings, torch.nn.Linear):
                with deepspeed.zero.GatheredParameters(
                    output_embeddings.weight, modifier_rank=0
                ):
                    desc_output_embed = output_embeddings.weight[desc_token_ids].mean(
                        dim=0
                    )
                    output_embeddings.weight[token_id] = desc_output_embed

                if output_embeddings.bias is not None:
                    with deepspeed.zero.GatheredParameters(
                        output_embeddings.bias, modifier_rank=0
                    ):
                        desc_output_bias = output_embeddings.bias[desc_token_ids].mean()
                        output_embeddings.bias[token_id] = desc_output_bias.item()


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig
) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
