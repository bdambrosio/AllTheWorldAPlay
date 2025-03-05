from google import genai
from google.genai import types # type: ignore
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, Response
import os
from PIL import Image
from io import BytesIO

import requests

app = FastAPI()

client = genai.Client(api_key='GEMINI_API_KEY')

prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    #processed_prompt = extract_key_elements(prompt)
    print(f"Original prompt: {prompt}")
    #print(f"Processed prompt: {processedprompt}")
    response = client.models.generate_images(
        model='imagen-3.0-generate-002',
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images= 1,
        )
    )

    if response.status_code == 200 and response.json().get('output'):
        for generated_image in response.generated_images:
            image = Image.open(BytesIO(generated_image.image.image_bytes))
            image.show()
            img_buffer = BytesIO(image.content)
            img_bytes = img_buffer.getvalue()
            return Response(content=img_bytes, media_type="image/png")
    else:
        print(f"Response is not a dict: {response}")
        return Response(content=response.content, media_type="application/json")
    
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    

