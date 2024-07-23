import cv2
import numpy as np
import argparse
import os


def load_image(image_path):
    """Load an image from a file path."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def apply_canny_edge_detection(image, threshold1, threshold2):
    """Apply Canny edge detection to an image."""
    return cv2.Canny(image, threshold1, threshold2)


def draw_wide_edges_on_image(original_image, edges, width):
    """Draw a wide path around the edges on the original image with grey color."""
    result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    # Dilate the edges to create a wider path
    kernel = np.ones((width, width), np.uint8)
    wide_edges = cv2.dilate(edges, kernel, iterations=1)
    result_image[wide_edges != 0] = [128, 128, 128]  # Drawing edges in grey
    return result_image


def save_image(image, output_path):
    """Save the image to a file."""
    cv2.imwrite(output_path, image)


def process_image(image_path, threshold1, threshold2, output_dir, width):
    image = load_image(image_path)
    if image is None:
        print(f"Error: Unable to load image from path: {image_path}")
        return

    edges = apply_canny_edge_detection(image, threshold1, threshold2)
    result_image = draw_wide_edges_on_image(image, edges, width)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    save_image(result_image, output_path)
    print(f"Result image saved to {output_path}")


def main(image_path, threshold1, threshold2, output_dir, width):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            file_path = os.path.join(image_path, filename)
            if os.path.isfile(file_path):
                process_image(file_path, threshold1, threshold2, output_dir, width)
    else:
        process_image(image_path, threshold1, threshold2, output_dir, width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply Canny edge detection to an image or a folder of images and save the result."
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the image file or folder."
    )
    parser.add_argument(
        "--threshold1",
        type=int,
        default=100,
        help="First threshold for the hysteresis procedure.",
    )
    parser.add_argument(
        "--threshold2",
        type=int,
        default=200,
        help="Second threshold for the hysteresis procedure.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the result images.",
    )
    parser.add_argument(
        "--width", type=int, default=20, help="Width of the path around the edges."
    )
    args = parser.parse_args()

    main(args.image_path, args.threshold1, args.threshold2, args.output_dir, args.width)
