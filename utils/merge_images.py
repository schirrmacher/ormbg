import os
import cv2
import argparse
import random
import string
import albumentations as A


def create_ground_truth_mask(image):
    return image[:, :, 3]


def create_random_filename_from_filepath(path):
    letters = string.ascii_lowercase
    random_string = "".join(random.choice(letters) for i in range(13))
    return random_string + "_" + os.path.basename(path)


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
    background_path, segmentation_path, image_path, ground_truth_path
):
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)

    if segmentation.shape[2] < 4:
        raise Exception(f"Image does not have an alpha channel: {segmentation_path}")

    file_name = create_random_filename_from_filepath(segmentation_path)
    image_path = os.path.join(image_path, file_name)
    ground_truth_path = os.path.join(ground_truth_path, file_name)

    ground_truth = create_ground_truth_mask(segmentation)
    result = merge_images(background, segmentation)

    assert ground_truth.shape[0] == result.shape[0]
    assert ground_truth.shape[1] == result.shape[1]

    cv2.imwrite(ground_truth_path, ground_truth)
    cv2.imwrite(image_path, result)


def main():
    parser = argparse.ArgumentParser(
        description="Merge two images with one image having transparency."
    )
    parser.add_argument(
        "-b", "--background", required=True, help="Path to the background image"
    )
    parser.add_argument(
        "-s", "--segmentation", required=True, help="Path to the segmentation image"
    )
    parser.add_argument(
        "-im",
        "--image-path",
        type=str,
        default="im",
        help="Path where the merged image will be saved",
    )
    parser.add_argument(
        "-gt",
        "--groundtruth-path",
        type=str,
        default="gt",
        help="Ground truth folder",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        os.makedirs(args.image_path)
    if not os.path.exists(args.groundtruth_path):
        os.makedirs(args.groundtruth_path)

    create_training_data(
        background_path=args.background,
        segmentation_path=args.segmentation,
        image_path=args.image_path,
        ground_truth_path=args.groundtruth_path,
    )


if __name__ == "__main__":
    main()
