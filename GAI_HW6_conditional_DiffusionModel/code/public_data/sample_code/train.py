"""
Sample code to load the training data, pretrained module and training.
You can modify the code as your needs.
"""
from glob import glob
import json
import os
import random

from PIL import Image
import tensorboard
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusers.utils import WEIGHTS_NAME
from safetensors.torch import save_file, load_file
from torch.amp import autocast

from torch import amp
from torch.amp import autocast

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

# We provide a sample dataloader for this dataset, you can modify it as needed.
class TextImageDataset(Dataset):
    def __init__(self, data_root, caption_file, tokenizer, size=256):
        self.data_root = data_root
        self.tokenizer = tokenizer
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        self.image_files = glob(os.path.join(data_root, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]

        ## Load image
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)

        ## Load text prompt
        # Image file name: "mosterID_ACTION_frameID.png"
        # key in "train_ingo.json": "mosterID_ACTION"
        key = img_file.split("/")[-1].split(".")[0]
        key = "_".join(key.split("_")[:-1])

        # Sample caption =  moster description + action description
        given_descriptions = self.captions[key]['given_description']
        given_description = random.choice(given_descriptions)
        caption = f"{given_description} {self.captions[key]['action_description']}"
        caption = "" if random.random() < 0.1 else caption
        inputs = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids
        
        return {
            "pixel_values": image,
            "input_ids": inputs.squeeze(0),
        }


@torch.no_grad()
def generate_and_save_images(unet, vae, text_encoder, tokenizer, epoch, device, save_folder, guidance_scale=2, enable_bar=False):
    unet.eval()
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(50)  # Use fewer steps for faster preview

    test_prompts = [
        "A red tree monster with a skull face and twisted branches.",
        "Blood-toothed monster with spiked fur, wielding an axe, and wearing armor. The monster is moving.",
        "Gray vulture monster with wings, sharp beak, and trident.",
        "Small, purple fish-like creature with large eye and pink fins. The monster is being hit.",
    ]
    batch_size = 1
    for i, prompt in enumerate(test_prompts):
        batch_size = 1

        # 條件輸入 (有文字描述)
        cond_input = tokenizer([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
        cond_emb = text_encoder(cond_input)[0]

        # 無條件輸入（空文字）
        uncond_input = tokenizer([""], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
        uncond_emb = text_encoder(uncond_input)[0]

        # 初始潛在變數
        latents = torch.randn((batch_size, 4, 32, 32)).to(device)

        for t in scheduler.timesteps:
            t = t.to(device)
            latent_input = latents

            # 預測 noise（有條件 & 無條件）
            noise_pred_cond = unet(latent_input, t, encoder_hidden_states=cond_emb).sample
            noise_pred_uncond = unet(latent_input, t, encoder_hidden_states=uncond_emb).sample

            # Classifier-Free Guidance（CFG）
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 更新 latent
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 解碼
        latents = latents / 0.18215
        images = vae.decode(latents, return_dict=False)[0]
        images = (images.clamp(-1, 1) + 1) / 2  # [-1, 1] → [0, 1]

        # 儲存圖片
        to_pil = transforms.ToPILImage()
        image_pil = to_pil(images[0].cpu())
        image_pil.save(os.path.join(save_folder, f"epoch_sch_{epoch:03d}_{i}.png"))

    unet.train()

def train():
    # ========= Hyperparameters ==========
    train_epochs = 10000
    batch_size = 32
    gradient_accumulation_steps = 16 # 1 
    # You can use gradients accumulation to simulate larger batch size if you have limited GPU memory.
    # Call optimizer.step() every `gradient_accumulation_steps` batches.
    lr = 1e-5
    eval_freq = 500
    save_freq = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Saved folders ==========
    ckpt_folder = "outputs/ckpt_last"
    save_folder = "outputs/samples_last"
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    # ========== Load Pretrained Model ==========
    ## You cannot change this part
    pretrain_CLIP_path = "openai/clip-vit-base-patch32"
    pretrain_VAE_path = "CompVis/stable-diffusion-v1-4"

    # Load pre-trained CLIP tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(pretrain_CLIP_path)
    text_encoder = CLIPTextModel.from_pretrained(pretrain_CLIP_path).eval().to(device)
    text_encoder.requires_grad_(False)

    # Load pre-trained VAE
    vae = AutoencoderKL.from_pretrained(pretrain_VAE_path, subfolder="vae").to(device)
    vae.requires_grad_(False)

    # ========== Init ==========
    ## You should modify the model architecture by your self
    """
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(160),
        down_block_types=("CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D"),
        cross_attention_dim=512,
    ).to(device)
    unet.train()
    optimizer = torch.optim.Adam(list(unet.parameters()), lr=lr)
    noise_scheduler = DDPMScheduler()
    """
    
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        block_out_channels=[256, 384, 512, 768],
        layers_per_block=2,
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ],
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ],
        cross_attention_dim=512,
        attention_head_dim=16
    ).to(device)
    """
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        block_out_channels=[256, 384, 512, 768],  # 注意力 + 階層解像
        down_block_types=[
            "CrossAttnDownBlock2D",  # Stage 1
            "CrossAttnDownBlock2D",  # Stage 2
            "CrossAttnDownBlock2D",  # Stage 3
            "DownBlock2D",           # Stage 4 不加 attention 省記憶體
        ],
        up_block_types=[
            "UpBlock2D",             # Stage 1 不加 attention
            "CrossAttnUpBlock2D",    # Stage 2
            "CrossAttnUpBlock2D",    # Stage 3
            "CrossAttnUpBlock2D",    # Stage 4
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",  # 中間保留 Cross Attention
        cross_attention_dim=768,                   # 和 CLIP encoder 對齊
        transformer_layers_per_block=(1, 1, 2, 2), # attention 層數遞增（越深越 expressive）
        attention_head_dim=32,                     # 每個 head 維度
        dropout=0.1,                               # 提升泛化力
        norm_num_groups=32,                        # 預設 group norm
        flip_sin_to_cos=True,                      # 時間 embedding 穩定性
        time_embedding_type="positional",          # 可改 fourier 試試
    ).to(device)
    """
    unet.train()
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    noise_scheduler = DDPMScheduler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    ckpt_path = "/mnt/shuof/HW6/public_data/sample_code/outputs/ckpt_last/unet_sch_38137"
    if os.path.exists(ckpt_path):
        print(f"🔄 載入 checkpoint: {ckpt_path}")
        unet.load_state_dict(load_file(os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")))
        print("✅ 權重載入成功！")

    # ========== Dataset ==========
    dataset = TextImageDataset("/mnt/shuof/HW6/public_data/train", "/mnt/shuof/HW6/public_data/train_info.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(dataset)}")
    print("Number of batches:", len(dataloader))
    # Test the generation pipeline
    generate_and_save_images(unet, vae, text_encoder, tokenizer, 0, device, save_folder)

    # ========== Training ==========
    loss_accumulated = 0.0
    step = 38138
    scaler = amp.GradScaler("cuda")

    for epoch in range(train_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        optimizer.zero_grad()
        epoch_loss = 0.0

        for idx, batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode text and images
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast(device_type='cuda', dtype=torch.bfloat16):  # 混合精度前向傳播
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / gradient_accumulation_steps  # 平均化 loss

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

                pbar.set_description(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

                if step % eval_freq == 0:
                    generate_and_save_images(unet, vae, text_encoder, tokenizer, step, device, save_folder)

                if step % save_freq == 0:
                    unet.save_pretrained(os.path.join(ckpt_folder, f"unet_sch_{step}"))

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()
