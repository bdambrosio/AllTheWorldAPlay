from fastapi import FastAPI, Response
from diffusers import DiffusionPipeline
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from io import BytesIO

app = FastAPI()

import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler
from peft import PeftModel

# Load LoRA
transformer = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="transformer",
    torch_dtype=torch.float16,
)
transformer = PeftModel.from_pretrained(transformer, "jasperai/flash-sd3")


# Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    transformer=transformer,
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None
)

# Scheduler
pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
  "stabilityai/stable-diffusion-3-medium-diffusers",
  subfolder="scheduler",
)

pipe.to("cuda")

prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."

image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    #pipeline("An image of a squirrel in Picasso style").images[0]
    image = pipe(prompt=prompt, num_inference_steps=40, guidance_scale=0.0).images[0]
    
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    

