from fastapi import FastAPI, Response
import torch
from diffusers import IFPipeline, IFSuperResolutionPipeline # type: ignore
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load base model for initial generation
pipe = IFPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

# Load super-resolution model
pipe_super = IFSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "512x512"):
    print(f"Generating image for prompt: {prompt}")
    
    # Generate initial image with text embeddings
    prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
    
    # Stage 1
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_inference_steps=25,
        height=256,
        width=256
    ).images[0]
    
    # Stage 2 - Upscale
    image = pipe_super(
        image=image,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_inference_steps=25,
        height=512,
        width=512
    ).images[0]
    
    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    