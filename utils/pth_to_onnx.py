import torch
import argparse
from ormbg.models.ormbg import ORMBG


def export_to_onnx(model_path, onnx_path):

    net = ORMBG()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    net.eval()

    # Create a dummy input tensor. The size should match the model's input size.
    # Adjust the dimensions as necessary; here it is assumed the input is a 3-channel image.
    dummy_input = torch.randn(
        1,
        3,
        1024,
        1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a trained model to ONNX format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ormbg.pth",
        help="The path to the trained model file.",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="models/ormbg.pth",
        help="The path where the ONNX model will be saved.",
    )

    args = parser.parse_args()

    export_to_onnx(args.model_path, args.onnx_path)
