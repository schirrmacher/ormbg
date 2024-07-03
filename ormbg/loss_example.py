import os
import torch
import argparse
import numpy as np
from skimage import io
import skimage.transform as transform
from models.ormbg import ORMBG
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove background from images using ORMBG model."
    )
    parser.add_argument(
        "--prediction",
        type=list,
        default=[
            os.path.join("examples", "loss", "loss01.png"),
            os.path.join("examples", "loss", "loss02.png"),
            os.path.join("examples", "loss", "loss03.png"),
            os.path.join("examples", "loss", "loss04.png"),
            os.path.join("examples", "loss", "loss05.png"),
        ],
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=os.path.join("examples", "loss", "gt.png"),
        help="Ground truth mask",
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


def inference(args):
    prediction_paths = args.prediction
    gt_path = args.gt

    net = ORMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pred_path in prediction_paths:

        model_input_size = [1024, 1024]
        loss = io.imread(pred_path)
        prediction = preprocess_image(loss, model_input_size).to(device)

        model_input_size = [1024, 1024]
        gt = io.imread(gt_path)
        ground_truth = preprocess_image(gt, model_input_size).to(device)

        _, loss = net.compute_loss([prediction], ground_truth)

        print(f"Loss: {pred_path} {loss}")


if __name__ == "__main__":
    inference(parse_args())
