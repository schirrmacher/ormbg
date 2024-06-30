import os
import argparse
from PIL import Image


def convert_rgba_to_rgb(input_path, output_path):
    # Open the input image
    rgba_image = Image.open(input_path)

    # Convert the RGBA image to RGB by removing the alpha channel
    rgb_image = rgba_image.convert("RGB")

    # Save the resulting RGB image
    rgb_image.save(output_path)

    print(f"Converted image saved to {output_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert an RGBA image to an RGB image."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("examples", "image01_depth.png"),
        help="Path to the input RGBA image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("examples", "image01_no_depth.png"),
        help="Path to save the output RGB image",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the image
    convert_rgba_to_rgb(args.input, args.output)


if __name__ == "__main__":
    main()
