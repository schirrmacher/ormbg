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

    if bh != fh or bw != fw:
        background = cv2.resize(background, (fw, fh), interpolation=cv2.INTER_AREA)
    return background


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


def create_training_data(
    background_dir, segmentation_dir, image_path, ground_truth_path
):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)

    background_files = [
        os.path.join(background_dir, f)
        for f in os.listdir(background_dir)
        if os.path.isfile(os.path.join(background_dir, f))
    ]
    segmentation_files = [
        os.path.join(segmentation_dir, f)
        for f in os.listdir(segmentation_dir)
        if os.path.isfile(os.path.join(segmentation_dir, f))
    ]

    for segmentation_path in segmentation_files:
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        if segmentation.shape[2] < 4:
            raise Exception(
                f"Image does not have an alpha channel: {segmentation_path}"
            )

        background_path = random.choice(background_files)
        background = cv2.imread(background_path, cv2.IMREAD_COLOR)

        background = resize_background_if_needed(background, segmentation)

        file_name = create_random_filename_from_filepath(segmentation_path)
        image_output_path = os.path.join(image_path, file_name)
        ground_truth_output_path = os.path.join(ground_truth_path, file_name)

        ground_truth = create_ground_truth_mask(segmentation)
        result = merge_images(background, segmentation)

        assert ground_truth.shape[0] == result.shape[0]
        assert ground_truth.shape[1] == result.shape[1]

        cv2.imwrite(ground_truth_output_path, ground_truth)
        cv2.imwrite(image_output_path, result)


def main():
    parser = argparse.ArgumentParser(
        description="Merge images in folders with one image having transparency."
    )
    parser.add_argument(
        "-bd",
        "--background-dir",
        required=True,
        help="Path to the background images directory",
    )
    parser.add_argument(
        "-sd",
        "--segmentation-dir",
        required=True,
        help="Path to the segmentation images directory",
    )
    parser.add_argument(
        "-im",
        "--image-path",
        type=str,
        default="im",
        help="Path where the merged images will be saved",
    )
    parser.add_argument(
        "-gt",
        "--groundtruth-path",
        type=str,
        default="gt",
        help="Ground truth folder",
    )
    args = parser.parse_args()

    create_training_data(
        background_dir=args.background_dir,
        segmentation_dir=args.segmentation_dir,
        image_path=args.image_path,
        ground_truth_path=args.groundtruth_path,
    )


if __name__ == "__main__":
    main()
