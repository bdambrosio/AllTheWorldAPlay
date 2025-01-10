from fastapi import FastAPI, Response
import torch
from diffusers import StableDiffusion3Pipeline # type: ignore

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16
).to("cuda")

#pipe = StableDiffusion3Pipeline.from_pretrained(
#    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
#).to("cuda")
from PIL import Image
from io import BytesIO

app = FastAPI()

pipe.to("cuda")

prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."

def extract_key_elements(prompt: str, max_length: int = 77) -> str:
    """Convert narrative prompt into keyword string"""
    # Common non-descriptive words to remove
    skip_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'will', 'would',
                 'have', 'has', 'had', 'be', 'been', 'being', 'to', 'for', 'of',
                 'with', 'by', 'at', 'in', 'on', 'you', 'i', 'he', 'she', 'it',
                 'we', 'they', 'this', 'that', 'these', 'those', 'now', 'then',
                 'and', 'or', 'but', 'so', 'because', 'if', 'when', 'where',
                 'what', 'who', 'how', 'why', 'which'}

    # Split into words and filter
    words = prompt.replace('.', ' ').replace(',', ' ').lower().split()
    keywords = [word for word in words if word not in skip_words]
    
    # Always include style at start
    if 'photorealistic' not in keywords:
        keywords.insert(0, 'photorealistic')
        
    result = ' '.join(keywords)
    
    print(f"\nKeyword Extraction:")
    print(f"Original ({len(prompt)} chars): {prompt}")
    print(f"Keywords ({len(result)} chars): {result}\n")
    
    return result[:max_length]

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    #processed_prompt = extract_key_elements(prompt)
    print(f"Original prompt: {prompt}")
    #print(f"Processed prompt: {processedprompt}")
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
    

