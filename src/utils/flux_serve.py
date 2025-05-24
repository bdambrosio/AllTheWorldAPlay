import time
from diffusers import DiffusionPipeline
import torch
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, Response

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell",
                                         torch_dtype=torch.bfloat16,
                                         device_map='balanced')


app = FastAPI()

prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X256"):
    #processed_prompt = extract_key_elements(prompt)
    print(f"Original prompt: {prompt}")
    #print(f"Processed prompt: {processedprompt}")
    image = pipe(prompt=prompt, 
                 height=512,
                 width=512,
                 num_inference_steps=20
                 ).images[0]
    
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")
