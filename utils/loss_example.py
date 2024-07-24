import os
import torch
import argparse
import numpy as np
from skimage import io
import torch.nn.functional as F

import sys

sys.path.append("../ormbg")
sys.path.append("../experiments")

from ormbg.models.ormbg import ORMBG
from experiments.loss import PixLoss, ClsLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove background from images using ORMBG model."
    )
    parser.add_argument(
        "--prediction",
        type=list,
        default=[
            os.path.join("examples", "loss", "gt.png"),
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

    birefnet_pix_loss_calculation = PixLoss()
    birefnet_cls_loss_calculation = ClsLoss()

    for prediction_path in prediction_paths:

        model_input_size = [1024, 1024]
        prediction_image = io.imread(prediction_path)
        prediction = preprocess_image(prediction_image, model_input_size).to(device)

        model_input_size = [1024, 1024]
        ground_truth_image = io.imread(gt_path)
        ground_truth = preprocess_image(ground_truth_image, model_input_size).to(device)

        _, is_net_loss = net.compute_loss([prediction], ground_truth)
        birefnet_pix_loss = birefnet_pix_loss_calculation.forward(
            [prediction], ground_truth
        )
        birefnet_cls_loss = birefnet_cls_loss_calculation.forward(
            [prediction], ground_truth
        )
        print(
            f"Loss: {prediction_path} {is_net_loss} {birefnet_pix_loss} {birefnet_cls_loss}"
        )


if __name__ == "__main__":
    inference(parse_args())
