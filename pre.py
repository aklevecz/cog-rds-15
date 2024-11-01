# from diffusers import StableDiffusionPipeline
# import torch

# # Load your fine-tuned model
# pipeline = StableDiffusionPipeline.from_pretrained(
#     "cyburn/midjourney_v4_finetune",  # Local path or HF repo name if uploaded
#     torch_dtype=torch.float16,  # Use float16 for better memory efficiency
#     safety_checker=None  # Optional: disable safety checker if needed
# )

# # Move to GPU if available
# pipeline = pipeline.to("cuda")

# # Generate an image
# image = pipeline(
#     prompt="your prompt here",
#     num_inference_steps=30,
#     guidance_scale=7.5
# ).images[0]

# # Save the image
# image.save("generated_image.png")
from diffusers import StableDiffusionPipeline
import torch

# First check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model with the correct dtype and device
pipe = StableDiffusionPipeline.from_pretrained(
    "wyyadd/sd-1.5",
    torch_dtype=torch.float16
)

# Move the pipeline to GPU first
pipe = pipe.to(device)

# Then load the LORA weights
pipe.load_lora_weights("rds-15-lora.safetensors")

# Generate the image
prompt = "rdsflower"
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("pokemon.png")