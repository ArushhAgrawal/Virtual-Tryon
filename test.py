import os
import gc
import torch
from PIL import Image
from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline

# Environment & Device Configuration
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = "mps" if torch.backends.mps.is_available() else "cpu"
weight_dtype = torch.float16 if device == "mps" else torch.float32
print(f"Device: {device} | Precision: {weight_dtype}")

# Local Paths
BASE_SD_PATH = "runwayml/stable-diffusion-inpainting"
CATVTON_PATH = "./checkpoints/CatVTON_512"
SCHP_PATH = "./checkpoints/schp"

# Load Models
print("Loading CatVTON Pipeline...")
pipeline = CatVTONPipeline(
    base_ckpt=BASE_SD_PATH,
    attn_ckpt=CATVTON_PATH,
    attn_ckpt_version="vitonhd",
    weight_dtype=weight_dtype,
    device=device,
    skip_safety_check=True
)

print("Loading AutoMasker...")
automasker = AutoMasker(
    densepose_ckpt=None,
    schp_ckpt=SCHP_PATH,
    device="cpu"
)

def run_tryon(person_path, shirt_path, output_path="output.png"):
    print(f"Loading and resizing images to 384x512...")
    person_img = Image.open(person_path).convert("RGB").resize((384, 512), Image.BILINEAR)
    shirt_img = Image.open(shirt_path).convert("RGB").resize((384, 512), Image.BILINEAR)

    
# 1. Generate mask and parse maps
    parse_output = automasker(person_img, mask_type="upper")
    mask = parse_output["mask"]
    lip = parse_output["lip"]

# 2. Run diffusion inpainting
    with torch.inference_mode():
        diffusion_output = pipeline(
        image=person_img,
        condition_image=shirt_img,
        mask=mask,
        num_inference_steps=20,
        guidance_scale=2.5,
        height=512,
        width=384,
        generator=torch.Generator(device=device).manual_seed(42)
    )[0]

# 3. Lock original face details over the result
    final_result = AutoMasker.restore_identity(original_img=person_img,
            generated_img=diffusion_output,
            lip=lip,
            blur_kernel=(11,11))

    final_result.save("output.png")

    print(f"Complete! Result saved to {output_path}")

    gc.collect()

if __name__ == "__main__":
    run_tryon("data/person.jpg", "data/shirt.jpg")
