import os
import argparse
from PIL import Image
from typing import List


def convert_images(folder_path: str, direction: str) -> None:
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    files: List[str] = os.listdir(folder_path)

    # Filter files based on the conversion direction
    if direction == "to_png":
        files_to_convert = [
            file for file in files if file.lower().endswith((".jpg", ".jpeg"))
        ]
    elif direction == "to_jpg":
        files_to_convert = [file for file in files if file.lower().endswith(".png")]

    total_files = len(files_to_convert)
    converted_count = 0

    for file in files_to_convert:
        if direction == "to_png":
            file_path: str = os.path.join(folder_path, file)
            img: Image.Image = Image.open(fp=file_path)
            output_file_path: str = os.path.splitext(p=file_path)[0] + ".png"
            img.save(fp=output_file_path, format="PNG")
            converted_count += 1
            print(f"Converted {file} to PNG. [{converted_count}/{total_files}]")
        elif direction == "to_jpg":
            file_path: str = os.path.join(folder_path, file)
            img: Image.Image = Image.open(fp=file_path)
            output_file_path: str = os.path.splitext(p=file_path)[0] + ".jpg"
            img = img.convert(mode="RGB")  # Convert to RGB to ensure compatibility
            img.save(fp=output_file_path, format="JPEG")
            converted_count += 1
            print(f"Converted {file} to JPG. [{converted_count}/{total_files}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert images between JPG and PNG formats in a specified folder."
    )
    parser.add_argument(
        "direction",
        type=str,
        choices=["to_png", "to_jpg"],
        help='The direction of conversion: "to_png" or "to_jpg".',
    )
    parser.add_argument(
        "folder_path", type=str, help="The path to the folder containing the images."
    )

    args = parser.parse_args()
    convert_images(folder_path=args.folder_path, direction=args.direction)
