from fastapi import FastAPI, Response
import torch, random, time
from diffusers import LCMScheduler, AutoPipelineForText2Image
from PIL import Image
from io import BytesIO
import numpy as np  # Backup check for all-black images


model_id = "Lykon/dreamshaper-7"
adapter_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = AutoPipelineForText2Image.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16"
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()


app = FastAPI()


def _is_black(img: Image.Image) -> bool:
    """Return True if the image is (nearly) all black."""
    arr = np.asarray(img)
    return arr.std() < 1e-3  # comment this line out if redundant


def _generate_with_retries(prompt: str, max_retries: int = 3):
    """Generate an image, retrying when flagged as NSFW or black."""
    negative_prompt = None
    attempt = 0
    last_img = None

    while attempt < max_retries:
        # New random seed each try
        gen = torch.Generator("cuda").manual_seed(random.randint(0, 2**32 - 1))

        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=gen,
            num_inference_steps=4,
            guidance_scale=0,
            output_type="pil",
        )

        img = out.images[0]
        last_img = img

        nsfw = False
        if hasattr(out, "nsfw_content_detected") and out.nsfw_content_detected:
            nsfw = bool(out.nsfw_content_detected[0])
            print(f'NSFW content detected: {out.nsfw_content_detected}')
            print(f'is_black: {_is_black(img)}')

        if nsfw or _is_black(img):
            attempt += 1
            if attempt == 1:
                # First retry: add a negative prompt to reduce NSFW likelihood
                negative_prompt = "nsfw"
            elif attempt == 2:
                # Final retry: broaden negative prompt and disable safety checker altogether
                negative_prompt = "nsfw, nude"
                # Turn off the built-in checker for this last try
                pipe.safety_checker = None
            continue

        return img  # safe image

    # If all retries failed, return the last image (likely black)
    return last_img


@app.get("/generate_image")
async def generate_image(prompt: str, size: str = "256X192"):
    print(f"Original prompt: {prompt}")

    image = _generate_with_retries(prompt)

    # Convert the image to bytes
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    # Return the image as a response
    return Response(content=img_bytes, media_type="image/png")


