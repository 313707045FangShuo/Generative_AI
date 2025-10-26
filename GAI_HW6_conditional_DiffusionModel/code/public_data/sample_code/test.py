import json, os
from PIL import Image
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

stage = "sch_46000"
set_timesteps = 50

@torch.no_grad()
def generate(prompt, unet, tokenizer, text_encoder, vae, scheduler, device, guidance_scale=7.5):
    # 1. Encode text
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    input_ids = text_input.input_ids.to(device)
    encoder_hidden_states = text_encoder(input_ids)[0]

    # 2. Prepare unconditional embedding for CFG
    uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # 3. Concatenate for classifier-free guidance
    encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states], dim=0)

    # 4. Prepare latent noise
    latents = torch.randn((1, unet.config.in_channels, 32, 32), device=device)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(set_timesteps)

    # 5. Denoising loop
    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)  # for CFG
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample

        # CFG: classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 6. Decode latent to image
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image.clamp(-1, 1) + 1) / 2
    image = image.cpu()
    return image

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained modules
    pretrained_clip = "openai/clip-vit-base-patch32"
    pretrained_vae  = "CompVis/stable-diffusion-v1-4"

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_clip)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_clip).to(device).eval()
    text_encoder.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(pretrained_vae, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)

    # Load trained UNet
    unet_path = f"./outputs/ckpt_last/unet_{stage}"  # ← 改成你訓練完儲存的目錄
    unet = UNet2DConditionModel.from_pretrained(unet_path).to(device).eval()
    unet.requires_grad_(False)

    # Use DDIM scheduler for fast generation
    scheduler = DDIMScheduler.from_pretrained(pretrained_vae, subfolder="scheduler")

    # Read test prompts
    with open("test.json", "r") as f:
        test_data = json.load(f)

    save_folder = f"./outputs/png_{stage}_{set_timesteps}/"
    os.makedirs(save_folder, exist_ok=True)

    for key, value in tqdm(test_data.items()):
        text_prompt = value["text_prompt"]
        image_name = value["image_name"]

        image = generate(
            prompt=text_prompt,
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            device=device,
            guidance_scale=7.5,
        )

        save_image(image, os.path.join(save_folder, image_name))

if __name__ == "__main__":
    test()