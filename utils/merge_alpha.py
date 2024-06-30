import argparse
import os
from PIL import Image, ImageOps


def add_alpha_channel(original_path, alpha_path, output_path):
    original_image = Image.open(original_path)
    original_image = original_image.convert("RGBA")
    alpha_image = Image.open(alpha_path)
    alpha_image = alpha_image.convert("L")
    alpha_image = alpha_image.resize(original_image.size)
    r, g, b, _ = original_image.split()
    new_image = Image.merge("RGBA", (r, g, b, alpha_image))
    new_image.save(output_path)
    print(
        f"The image has been saved with the new (inverted) alpha channel at {output_path}."
    )


def process_folders(original_folder, alpha_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    original_files = os.listdir(original_folder)
    alpha_files = os.listdir(alpha_folder)

    alpha_dict = {
        os.path.splitext(f)[0]: os.path.join(alpha_folder, f) for f in alpha_files
    }

    for original_file in original_files:
        # Get the base name of the original file (without extension)
        base_name = os.path.splitext(original_file)[0]

        if base_name in alpha_dict:
            original_path = os.path.join(original_folder, original_file)
            alpha_path = alpha_dict[base_name]
            output_path = os.path.join(output_folder, f"{base_name}.png")
            add_alpha_channel(original_path, alpha_path, output_path)
        else:
            print(f"Alpha channel file not found for {original_file}. Skipping.")


def main():
    parser = argparse.ArgumentParser(
        description="Add an inverted alpha channel to images in a folder."
    )
    parser.add_argument(
        "original_folder",
        type=str,
        help="Path to the folder containing the original images.",
    )
    parser.add_argument(
        "alpha_folder",
        type=str,
        help="Path to the folder containing the black and white images to be used as the alpha channels.",
    )
    parser.add_argument(
        "output_folder", type=str, help="Path to save the output images."
    )

    args = parser.parse_args()

    process_folders(args.original_folder, args.alpha_folder, args.output_folder)


if __name__ == "__main__":
    main()
