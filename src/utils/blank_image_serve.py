from fastapi import FastAPI, Response
import os
from PIL import Image
from io import BytesIO
import base64

import requests

app = FastAPI()


# This is a 1x1 transparent PNG in base64
MINI_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    # Parse the size string (e.g., "256X192" -> (256, 192))
    width, height = map(int, size.upper().split('X'))
    
    # Create a new grey image
    img = Image.new('RGB', (width, height), color='#808080')  # #808080 is medium grey
    
    # Convert to bytes
    img_buffer = BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    

