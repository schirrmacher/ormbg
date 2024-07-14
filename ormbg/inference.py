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
        "-i",
        "--image",
        type=str,
        default=None,
        help="Path to the input image file or folder. If a folder is specified, all images in the folder will be processed.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="inference",
        help="Path to the output image file or folder. If a folder is specified, results will be saved in the specified folder.",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default=os.path.join("models", "ormbg.pth"),
        help="Path to the model file.",
    )
    parser.add_argument(
        "-c",
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


def process_image(image_path, output_path, model, device, model_input_size, compare):
    orig_im = io.imread(image_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    result = model(image)

    result_image = postprocess_image(result[0][0], orig_im_size)

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
        combined_image.save(output_path)
    else:
        no_bg_image.save(output_path)


def inference(args):
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

    if os.path.isdir(args.image):
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        image_files = [
            f
            for f in os.listdir(args.image)
            if os.path.isfile(os.path.join(args.image, f))
        ]
        total_images = len(image_files)

        for idx, file_name in enumerate(image_files):
            image_path = os.path.join(args.image, file_name)
            output_path = os.path.join(args.output, file_name)
            process_image(
                image_path, output_path, net, device, model_input_size, compare
            )
            print(f"Processed {idx + 1}/{total_images} images")
    else:
        process_image(args.image, args.output, net, device, model_input_size, compare)


if __name__ == "__main__":
    inference(parse_args())
