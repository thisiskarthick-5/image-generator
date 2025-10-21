import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
import random

# ---------- GPU setup ----------
torch.cuda.empty_cache()

# ---------- Load model (SDXL = very high quality) ----------
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# ---------- Automatic prompt enhancer ----------
def enhance_prompt(user_prompt):
    # Default subjects to use when prompt is meaningless
    default_subjects = [
        "a beautiful girl with flowers",
        "a futuristic city skyline",
        "a majestic mountain landscape",
        "a cozy room interior with warm lighting",
        "a cute puppy in a garden",
        "a cyberpunk street at night"
    ]
    # If the prompt is empty or nonsense, pick a random default
    if not user_prompt.strip() or len(user_prompt.strip()) < 3:
        user_prompt = random.choice(default_subjects)

    # Enhance it with professional descriptive terms
    enhancement = (
        "ultra realistic, high detail, 8k resolution, cinematic lighting, sharp focus, "
        "natural color tones, smooth textures, photorealistic quality, aesthetic composition"
    )
    return f"{user_prompt}, {enhancement}"

# ---------- Negative prompt (things to avoid) ----------
negative_prompt = (
    "blurry, distorted, bad anatomy, extra limbs, low quality, deformed face, "
    "dark, grainy, watermark, text, low detail, pixelated"
)

# ---------- Ask user ----------
user_prompt = input("Enter any prompt (even a dummy one): ")

# ---------- Generate final enhanced prompt ----------
final_prompt = enhance_prompt(user_prompt)

print("\n🧠 Enhanced prompt used by AI:")
print(final_prompt)
print("\n🎨 Generating image... please wait...")

# ---------- Image generation ----------
image = pipe(
    final_prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=768,
    guidance_scale=9.0,
    num_inference_steps=35
).images[0]

# ---------- Show image ----------
plt.imshow(image)
plt.axis('off')
plt.show()
