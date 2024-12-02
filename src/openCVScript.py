import cv2
import numpy as np

def stitch_vertical(images):
    """
    Stitch multiple images vertically into a seamless panorama.

    :param images: List of images as numpy arrays
    :return: Stitched panorama image
    """
    if len(images) < 2:
        raise ValueError("At least two images are required to stitch.")

    # Initialize the stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    # Perform stitching
    status, panorama = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        raise Exception(f"Stitching failed with status code {status}")

    return panorama


def read_images(image_paths):
    """
    Load images from file paths.

    :param image_paths: List of paths to image files
    :return: List of images as numpy arrays
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        images.append(img)
    return images




if __name__ == "__main__":
    # Provide paths to the images you want to stitch
    image_paths = [
    "data/raw/flower/1.jpg",  # Replace with the actual paths to your images
    "data/raw/flower/2.jpg",
    "data/raw/flower/3.jpg",
    "data/raw/flower/4.jpg",
]

    try:
        # Load images
        images = read_images(image_paths)

        # Stitch images vertically
        panorama = stitch_vertical(images)

        # Display the result
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the result
        cv2.imwrite("vertical_panorama.jpg", panorama)
        print("Panorama saved as vertical_panorama.jpg")
    except Exception as e:
        print(f"Error: {e}")
