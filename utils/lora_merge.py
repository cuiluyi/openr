from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(
    "/data/cuiluyi/openr/ckpts/DeepSeek-R1-Distill-Qwen-1.5B/format-lora-r1_distill_data"
)

# load base model and resize token embeddings
base_model = AutoModelForCausalLM.from_pretrained(
    "/data/cuiluyi/resources/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/",
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
)
base_model.resize_token_embeddings(len(tokenizer))

# load and merge LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "/data/cuiluyi/openr/ckpts/DeepSeek-R1-Distill-Qwen-1.5B/format-lora-r1_distill_data",
)
model = model.merge_and_unload()

# save the merged model and tokenizer
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
