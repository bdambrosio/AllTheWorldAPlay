from fastapi import FastAPI, Response
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
).to("cuda")

#pipe = StableDiffusion3Pipeline.from_pretrained(
#    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
#).to("cuda")
from PIL import Image
from io import BytesIO

app = FastAPI()

pipe.to("cuda")

prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    #pipeline("An image of a squirrel in Picasso style").images[0]
    image = pipe(prompt=prompt, 
                 num_inference_steps=40,
                 height=512,
                 width=512,
                 guidance_scale=4.00).images[0]
    
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    

