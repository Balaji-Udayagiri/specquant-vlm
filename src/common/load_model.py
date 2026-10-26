"""Single place to load the target VLM. Every experiment imports this.

Day 1 task: make the baseline load and generate on one image + prompt.
"""
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_llava(model_name: str, dtype: str = "float16", device_map: str = "auto", quantization_config=None):
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor

def make_prompt(processor, question: str) -> str:
    # LLaVA chat format. Keep this identical across all experiments.
    return f"USER: <image>\n{question} ASSISTANT:"
