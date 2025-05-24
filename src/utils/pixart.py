import torch, os, time
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import bitsandbytes
from diffusers import HiDreamImagePipeline
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# 4-bit + no hidden-state/attention outputs → ~2 GB VRAM
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,      # fp16 saves a little over bf16 here
    #load_in_8bit=True,
    output_hidden_states=False,
    output_attentions=False,
    device_map="auto"
)

pipe = HiDreamImagePipeline.from_pretrained(
    "HiDream-ai/HiDream-I1-Fast",
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.float16,
    device_map="balanced"
)

# memory savers -------------------------------------------------------------
#pipe.enable_xformers_memory_efficient_attention()
#pipe.enable_attention_slicing()
#pipe.enable_vae_slicing()           # diffusers ≥ 0.27
# --------------------------------------------------------------------------
# Optional: inspect where things went
from accelerate import infer_auto_device_map, dispatch_model
print(pipe.device)                  # overall pipe has no single device
print(pipe.text_encoder_4.device)   # e.g. cuda:0
#print(pipe.unet.device)             # e.g. cuda:1


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 1k"
image   = pipe(prompt, height=256, width=256, num_images_per_prompt=1, num_inference_steps=25).images[0]
image.save("out.png")

image.show()
time.sleep(10)
