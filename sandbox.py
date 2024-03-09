import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use o ckpt correto para suas configurações de etapas!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cpu", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cpu"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16).to("cpu")

# Certifique-se de que o amostrador use etapas de tempo "trailing".
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

prompt = "A girl smiling"
output = "output.png"

print("Gerando imagem para: \"" + prompt + "\"")

# Certifique-se de usar as mesmas etapas de inferência do modelo carregado e CFG definido como 0.
pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0].save(output)


print("\nIMGAGEM GERADA EM: \"" + output + "\"")