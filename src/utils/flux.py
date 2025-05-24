import time
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell",
                                         torch_dtype=torch.bfloat16,
                                         device_map='balanced')

prompt = "A quiet temperate mixed forest clearing illuminated by soft early morning light, featuring sunlit grassy patches interspersed with apple trees bearing ripe early summer fruit under a clear sky. Samantha is Samantha, a healthy, attractive young woman. Samantha picks several ripe apples from the nearest tree and closely inspects them, finding them intact with no signs of rot, insects, or unusual marks. She gains fresh, edible fruit and momentarily feels a slight easing of tension, though underlying suspicion remains. The gentle rustling of leaves continues softly around her as the environment remains calm."


prompt = "Samantha, a healthy attractive, but nervous young woman, picking ripe apples from trees in a quiet temperate forest clearing. She holds several fresh apples in her hands, standing among sunlit grassy patches with apple trees. Soft early morning light, clear sky, peaceful forest setting with gentle rustling leaves."

image = pipe(prompt, height=512,width=512).images[0]
image.show()
time.sleep(10)
