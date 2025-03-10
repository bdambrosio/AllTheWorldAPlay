from fastapi import FastAPI, Response
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
from PIL import Image
from io import BytesIO



model_id = "Lykon/dreamshaper-7"
adapter_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()


prompt = "cute kitten playing with mom"

app = FastAPI()


@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    print(f"Original prompt: {prompt}")
    # disable guidance_scale by passing 0
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")


