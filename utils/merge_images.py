import os
import cv2
import argparse
import random
import string


def create_ground_truth_mask(image):
    return image[:, :, 3]


def create_random_filename_from_filepath(path):
    letters = string.ascii_lowercase
    random_string = "".join(random.choice(letters) for i in range(13))
    return random_string + "_" + os.path.basename(path)


def resize_background_if_needed(background, foreground):
    bh, bw = background.shape[:2]
    fh, fw = foreground.shape[:2]

    # Compute the scale factors
    scale_w = fw / bw
    scale_h = fh / bh
    scale = max(scale_w, scale_h)

    # Calculate the new dimensions while keeping aspect ratio
    new_w = int(bw * scale)
    new_h = int(bh * scale)

    # Resize the background
    background_resized = cv2.resize(
        background, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    # Calculate the top-left corner of the cropping area
    x_start = (new_w - fw) // 2
    y_start = (new_h - fh) // 2

    # Crop the resized background to match the foreground dimensions
    background_cropped = background_resized[
        y_start : y_start + fh, x_start : x_start + fw
    ]

    return background_cropped


def merge_images(background, foreground, position=(0, 0)):
    x, y = position

    fh, fw = foreground.shape[:2]

    if x + fw > background.shape[1]:
        fw = background.shape[1] - x
        foreground = foreground[:, :fw]
    if y + fh > background.shape[0]:
        fh = background.shape[0] - y
        foreground = foreground[:fh, :]

    # Region of Interest (ROI) in the background where the foreground will be placed
    roi = background[y : y + fh, x : x + fw]

    # Split the foreground image into its color and alpha channels
    foreground_color = foreground[:, :, :3]
    alpha = foreground[:, :, 3] / 255.0

    # Blend the images based on the alpha channel
    for c in range(0, 3):
        roi[:, :, c] = (1.0 - alpha) * roi[:, :, c] + alpha * foreground_color[:, :, c]

    # Place the modified ROI back into the original image
    background[y : y + fh, x : x + fw] = roi

    return background


def create_training_data(background_dir, png_dir, image_dir, ground_truth_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)

    background_files = [
        os.path.join(background_dir, f)
        for f in os.listdir(background_dir)
        if os.path.isfile(os.path.join(background_dir, f))
    ]
    png_files = [
        os.path.join(png_dir, f)
        for f in os.listdir(png_dir)
        if os.path.isfile(os.path.join(png_dir, f)) and f.lower().endswith(".png")
    ]

    i = 0
    for png_path in png_files:

        try:
            png = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            if png.shape[2] < 4:
                raise Exception(f"Image does not have an alpha channel: {png_path}")

            background_path = random.choice(background_files)
            background = cv2.imread(background_path, cv2.IMREAD_COLOR)

            background = resize_background_if_needed(background, png)

            file_name = create_random_filename_from_filepath(png_path)
            image_output_path = os.path.join(image_dir, file_name)
            ground_truth_output_path = os.path.join(ground_truth_dir, file_name)

            ground_truth = create_ground_truth_mask(png)
            result = merge_images(background, png)

            assert ground_truth.shape[0] == result.shape[0]
            assert ground_truth.shape[1] == result.shape[1]

            cv2.imwrite(ground_truth_output_path, ground_truth)
            cv2.imwrite(image_output_path, result)

            print(f"{i}/{len(png_files)}")
            i += 1

        except Exception as e:
            print(f"Skipping {png_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge images in folders with one image having transparency."
    )
    parser.add_argument(
        "-bd",
        "--background_dir",
        required=True,
        help="Path to the background images directory",
    )
    parser.add_argument(
        "-png",
        "--png_dir",
        required=True,
        help="Path to the segmentation images directory",
    )
    parser.add_argument(
        "-im",
        "--image_dir",
        type=str,
        default="im",
        help="Path where the merged images will be saved",
    )
    parser.add_argument(
        "-gt",
        "--ground_truth_dir",
        type=str,
        default="gt",
        help="Ground truth folder",
    )
    args = parser.parse_args()

    try:
        create_training_data(
            background_dir=args.background_dir,
            png_dir=args.png_dir,
            image_dir=args.image_dir,
            ground_truth_dir=args.ground_truth_dir,
        )
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    main()
