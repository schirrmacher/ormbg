# This script takes RGBA masks and exports the alpha channel as ground truth
# as another image.

import os
import argparse
from PIL import Image


def extract_alpha_channel(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            if img.mode == "RGBA":
                alpha = img.split()[-1]

                alpha_output_path = os.path.join(output_folder, f"{filename}")
                alpha.save(alpha_output_path)
                print(f"Saved alpha channel for {filename} to {alpha_output_path}")
            else:
                print(f"Image {filename} does not have an alpha channel.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract alpha channels from PNG images."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the input folder containing PNG images."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder where alpha channels will be saved.",
    )

    args = parser.parse_args()

    # Ensure the input and output folders are not the same
    if os.path.abspath(args.input_folder) == os.path.abspath(args.output_folder):
        print("Error: Input and output folders must be different.")
        return

    extract_alpha_channel(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
