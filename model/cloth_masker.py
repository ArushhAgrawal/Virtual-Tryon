import os
import cv2
import numpy as np
from PIL import Image
from model.SCHP import SCHP

class AutoMasker:

    def __init__(self, densepose_ckpt=None, schp_ckpt="./checkpoints/schp", device="cpu"):
        super().__init__()
        self.device = device
        lip_path = os.path.join(schp_ckpt, "exp-schp-201908261155-lip.pth")
        atr_path = os.path.join(schp_ckpt, "exp-schp-201908301523-atr.pth")
        self.schp_processor_lip = SCHP(ckpt_path=lip_path, device=device)
        self.schp_processor_atr = SCHP(ckpt_path=atr_path, device=device)

    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        parse_lip = self.schp_processor_lip(image)
        parse_atr = self.schp_processor_atr(image)
        return {"densepose": None, "lip": parse_lip, "atr": parse_atr}

    def cloth_agnostic_mask(self, densepose_mask, lip, atr, part="upper"):
        w, h = lip.size
        lip_arr = np.array(lip)
        atr_arr = np.array(atr)

        mask = np.zeros((h, w), dtype=np.uint8)
        if part == "upper":
            # LIP: 5=Upper-cloth, 6=Dress, 7=Coat | ATR: 4=Upper-cloth, 7=Outer
            mask[np.isin(lip_arr, [5, 6, 7])] = 255
            mask[np.isin(atr_arr, [4, 7])] = 255
        elif part == "lower":
            # LIP: 9=Pants, 12=Skirt | ATR: 5=Skirt, 6=Pants
            mask[np.isin(lip_arr, [9, 12])] = 255
            mask[np.isin(atr_arr, [5, 6])] = 255
        elif part == "overall":
            mask[np.isin(lip_arr, [5, 6, 7, 9, 10, 12])] = 255
            mask[np.isin(atr_arr, [4, 5, 6, 7])] = 255
        else:
            mask[np.isin(lip_arr, [5, 6, 7])] = 255
            mask[np.isin(atr_arr, [4, 7])] = 255

        # Dilation expands boundary by 15px so old collar edges don't leak
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        return Image.fromarray(mask, mode="L")

    def restore_identity(original_img, generated_img, lip, blur_kernel= (11,11)):
        """
        Alpha-blends original face over the AI try-on result:
        Final = (Original * Alpha) + (Generated * (1.0 - Alpha))
        """
        orig_np = np.array(original_img.convert("RGB"))
        gen_np = np.array(generated_img.convert("RGB"))

        lip_arr = np.array(lip)
                
        # 1. Isolate Head & Facial regions
        face_binary = np.isin(lip_arr, [1, 2, 4, 13]).astype(np.uint8) * 255
        
        # 2. Feather the edges to prevent harsh cutout seams
        feathered = cv2.GaussianBlur(face_binary, blur_kernel, 0)
                
        # 3. Normalize into continuous float alpha channel [0.0, 1.0]
        alpha = (feathered.astype(np.float32) / 255.0)[:, :, np.newaxis]

        composite = (orig_np * alpha + gen_np * (1.0 - alpha)).astype(np.uint8)
        return Image.fromarray(composite)

    def __call__(self, image, mask_type="upper"):
        results = self.preprocess_image(image)
        mask = self.cloth_agnostic_mask(
            results["densepose"],
            results["lip"],
            results["atr"],
            part=mask_type
        )
        
        
        return {
            "mask": mask,
            "densepose": None,
            "lip": results["lip"],
            "atr": results["atr"]
        }
