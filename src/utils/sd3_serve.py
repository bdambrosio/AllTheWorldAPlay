from fastapi import FastAPI, Response
import torch
from diffusers import StableDiffusion3Pipeline # type: ignore
import bitsandbytes as bnb


#pipe = StableDiffusion3Pipeline.from_pretrained(
#    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16,
#).to("cuda")

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16,variant="fp16",
    device_map='balanced'
)

#pipe = StableDiffusion3Pipeline.from_pretrained(
#    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
#).to("cuda")
from PIL import Image
from io import BytesIO

app = FastAPI()

#pipe.to("cuda")

prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    #processed_prompt = extract_key_elements(prompt)
    print(f"Original prompt: {prompt}")
    #print(f"Processed prompt: {processedprompt}")
    image = pipe(prompt=prompt, 
                 height=384,
                 width=384,
                 num_inference_steps=40,
                 guidance_scale=4.00).images[0]
    
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    image = pipe(prompt=prompt, 
                 num_inference_steps=40,
                 guidance_scale=7.00).images[0]
    
    image.save('test.png')
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.show()
