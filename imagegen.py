import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = (
    "a beautiful girl with flowers in her hair, wearing a white dress, "
    "standing in a sunny flower field, ultra realistic, high detail, "
    "soft lighting, 8k, cinematic, sharp focus"
)

negative_prompt = (
    "blurry, distorted face, bad anatomy, extra limbs, low quality, dark, low resolution, watermark, text"
)

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=768,  # standard good resolution
    height=768,
    guidance_scale=8.0,  # controls how strongly it follows your text
    num_inference_steps=30  # improves quality
).images[0]

plt.imshow(image)
plt.axis("off")
plt.show()
