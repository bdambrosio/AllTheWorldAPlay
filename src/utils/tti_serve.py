from fastapi import FastAPI, Response
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
from io import BytesIO

app = FastAPI()

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    image = pipe(prompt=prompt, size=size, num_inference_steps=3, guidance_scale=0.0).images[0]
    
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    
