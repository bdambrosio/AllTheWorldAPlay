from fastapi import FastAPI, Response
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from peft import PeftModel
from PIL import Image
from io import BytesIO

app = FastAPI()

adapter_id = "jasperai/flash-sdxl"

pipe = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  use_safetensors=True,
)

pipe.scheduler = LCMScheduler.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  subfolder="scheduler",
  timestep_spacing="trailing",
)
pipe.to("cuda")

# Fuse and load LoRA weights
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    image = pipe(prompt=prompt, num_inference_steps=6, guidance_scale=0.0).images[0]
    
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    
