import os
import argparse
import numpy as np
import PIL.Image
import torch
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

# Configuration
if torch.backends.mps.is_available() and not torch.cuda.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("cpu")
    print("MPS device detected. Using CPU as a fallback for full compatibility.")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1600"))
MODEL_ID = os.getenv("MODEL_ID", "hustvl/vitmatte-base-distinctions-646")

# Load the model and processor
processor = VitMatteImageProcessor.from_pretrained(MODEL_ID)
model = VitMatteForImageMatting.from_pretrained(MODEL_ID).to(device)


# Helper functions
def resize_image(image: PIL.Image.Image, max_size: int) -> PIL.Image.Image:
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, PIL.Image.Resampling.LANCZOS)
    return image


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    mask[mask < 128] = 0
    mask[mask > 0] = 1
    return mask


def update_trimap(foreground_mask: np.ndarray, unknown_mask: np.ndarray) -> np.ndarray:
    foreground = foreground_mask[:, :, 0]
    foreground = binarize_mask(foreground)

    unknown = unknown_mask[:, :, 0]
    unknown = binarize_mask(unknown)

    trimap = np.zeros_like(foreground)
    trimap[unknown > 0] = 128
    trimap[foreground > 0] = 255
    return trimap


def adjust_background_image(
    background_image: PIL.Image.Image, target_size: tuple[int, int]
) -> PIL.Image.Image:
    target_w, target_h = target_size
    bg_w, bg_h = background_image.size

    scale = max(target_w / bg_w, target_h / bg_h)
    new_bg_w = int(bg_w * scale)
    new_bg_h = int(bg_h * scale)
    background_image = background_image.resize(
        (new_bg_w, new_bg_h), PIL.Image.Resampling.LANCZOS
    )
    left = (new_bg_w - target_w) // 2
    top = (new_bg_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    background_image = background_image.crop((left, top, right, bottom))
    return background_image


def replace_background(
    image: PIL.Image.Image, alpha: np.ndarray, background_image: PIL.Image.Image | None
) -> PIL.Image.Image | None:
    if background_image is None:
        return None

    if image.mode != "RGB":
        raise ValueError("Image must be RGB.")

    background_image = background_image.convert("RGB")
    background_image = adjust_background_image(background_image, image.size)

    image = np.array(image).astype(float) / 255
    background_image = np.array(background_image).astype(float) / 255
    result = image * alpha[:, :, None] + background_image * (1 - alpha[:, :, None])
    result = (result * 255).astype(np.uint8)
    return PIL.Image.fromarray(result)


@torch.inference_mode()
def run(
    image: PIL.Image.Image,
    trimap: PIL.Image.Image,
    apply_background_replacement: bool,
    background_image: PIL.Image.Image | None,
) -> tuple[np.ndarray, PIL.Image.Image, PIL.Image.Image | None]:
    if image.size != trimap.size:
        raise ValueError("Image and trimap must have the same size.")
    if max(image.size) > MAX_IMAGE_SIZE:
        raise ValueError(
            f"Image size is too large. Max image size is {MAX_IMAGE_SIZE} pixels."
        )
    if image.mode != "RGB":
        raise ValueError("Image must be RGB.")
    if trimap.mode != "L":
        raise ValueError("Trimap must be grayscale.")

    pixel_values = (
        processor(images=image, trimaps=trimap, return_tensors="pt")
        .to(device)
        .pixel_values
    )
    out = model(pixel_values=pixel_values)
    alpha = out.alphas[0, 0].to("cpu").numpy()

    w, h = image.size
    alpha = alpha[:h, :w]

    foreground = np.array(image).astype(float) / 255 * alpha[:, :, None] + (
        1 - alpha[:, :, None]
    )
    foreground = (foreground * 255).astype(np.uint8)
    foreground = PIL.Image.fromarray(foreground)

    if apply_background_replacement:
        res_bg_replacement = replace_background(image, alpha, background_image)
    else:
        res_bg_replacement = None

    return alpha, foreground, res_bg_replacement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViTMatte Image Matting")
    parser.add_argument(
        "--image", required=True, type=str, help="Path to the input image"
    )
    parser.add_argument(
        "--trimap", required=True, type=str, help="Path to the trimap image"
    )
    parser.add_argument(
        "--background", type=str, help="Path to the background image (optional)"
    )
    parser.add_argument(
        "--output_alpha",
        type=str,
        default="alpha.png",
        help="Path to save the alpha output image",
    )
    parser.add_argument(
        "--output_foreground",
        type=str,
        default="foreground.png",
        help="Path to save the foreground output image",
    )
    parser.add_argument(
        "--output_background",
        type=str,
        default="background_replacement.png",
        help="Path to save the background replacement output image",
    )

    args = parser.parse_args()

    # Load and resize images
    image = PIL.Image.open(args.image)
    image = resize_image(image, MAX_IMAGE_SIZE)
    trimap = PIL.Image.open(args.trimap).convert("L")  # Ensure trimap is grayscale
    trimap = resize_image(trimap, MAX_IMAGE_SIZE)
    background_image = (
        PIL.Image.open(args.background).convert("RGB") if args.background else None
    )

    # Run matting
    alpha, foreground, background_replacement = run(
        image, trimap, args.background is not None, background_image
    )

    # Save the outputs
    alpha_image = PIL.Image.fromarray((alpha * 255).astype(np.uint8))
    alpha_image.save(args.output_alpha)
    foreground.save(args.output_foreground)
    if background_replacement:
        background_replacement.save(args.output_background)
