import os
import torch
import argparse
import numpy as np
from PIL import Image
from skimage import io
from models.ormbg import ORMBG
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove background from images using ORMBG model."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join("examples", "image", "image01.png"),
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("image01_no_background.png"),
        help="Path to the output image file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "ormbg.pth"),
        help="Path to the model file.",
    )
    parser.add_argument(
        "--compare",
        action="store_false",
        help="Flag to save the original and processed images side by side.",
    )
    return parser.parse_args()


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
    ).type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    return image


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


def inference(args):
    image_path = args.image
    result_name = args.output
    model_path = args.model_path
    compare = args.compare

    net = ORMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    model_input_size = [1024, 1024]
    orig_im = io.imread(image_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    result = net(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)

    if pil_im.mode == "RGBA":
        pil_im = pil_im.convert("RGB")

    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(image_path)
    no_bg_image.paste(orig_image, mask=pil_im)

    if compare:
        combined_width = orig_image.width + no_bg_image.width
        combined_image = Image.new("RGBA", (combined_width, orig_image.height))
        combined_image.paste(orig_image, (0, 0))
        combined_image.paste(no_bg_image, (orig_image.width, 0))
        stacked_output_path = os.path.splitext(result_name)[0] + ".png"
        combined_image.save(stacked_output_path)
    else:
        no_bg_image.save(result_name)


if __name__ == "__main__":
    inference(parse_args())
