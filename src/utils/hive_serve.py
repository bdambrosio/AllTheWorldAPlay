from fastapi import FastAPI, Response
import os
from PIL import Image
from io import BytesIO

import requests

app = FastAPI()


HIVE_API_KEY = None
try:
    HIVE_API_KEY = os.getenv('HIVE_API_KEY')
except Exception as e:
    print(f"Error getting Hive API key: {e}")
prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."
print(f"HIVE_API_KEY: {HIVE_API_KEY}")

@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    #processed_prompt = extract_key_elements(prompt)
    print(f"Original prompt: {prompt}")
    #print(f"Processed prompt: {processedprompt}")
    headers = {
        'authorization': f'Bearer {HIVE_API_KEY}',
        'Content-Type': 'application/json',
    }

    json_data = {
        'input': {
            'prompt': prompt,
            'image_size': { 'width': 1024, 'height': 1024},
            'num_inference_steps': 15,
            'num_images': 1,
            'seed': 67,
            'output_format': 'png'
        }
    }
    print(f"JSON data: {json_data}")
    response = requests.post('https://api.thehive.ai/api/v3/hive/flux-schnell-enhanced', headers=headers, json=json_data) 
    if response.status_code == 200 and response.json().get('output'):
        print(f"Response is a dict and has output: {response}")
        image_url = response.json()['output'][0]['url']
        image = requests.get(image_url)
        img_buffer = BytesIO(image.content)
        img_bytes = img_buffer.getvalue()
        return Response(content=img_bytes, media_type="image/png")
    else:
        print(f"Response is not a dict: {response}")
        return Response(content=response.content, media_type="application/json")
    
    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
    

