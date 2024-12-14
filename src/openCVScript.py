import cv2
import numpy as np

def poisson_blend(src, dst, mask, center):
    """
    Perform Poisson blending on the overlapping region.

    :param src: Source image (part to be blended)
    :param dst: Destination image (base image)
    :param mask: Binary mask of the region to blend
    :param center: Center of the blending region in the destination image
    :return: Blended image
    """
    return cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)


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
    # Manually apply Poisson blending to smooth transitions
    for i in range(len(images) - 1):
        src = images[i + 1]
        dst = panorama

        # Define an overlapping region (adjust coordinates as needed)
        overlap_start = max(0, dst.shape[0] - src.shape[0])  # Adjust as needed
        overlap_end = dst.shape[0]

        # Extract regions
        src_overlap = src[:overlap_end - overlap_start, :]
        dst_overlap = dst[overlap_start:overlap_end, :]

        # Create a binary mask for the source overlap
        mask = np.zeros_like(src_overlap, dtype=np.uint8)
        mask[:, :] = 255  # Entire overlap area is part of the mask

        # Center the blending area (adjust this for more precise control)
        center = (dst_overlap.shape[1] // 2, dst_overlap.shape[0] // 2)

        # Blend the overlapping regions
        blended = poisson_blend(src_overlap, dst_overlap, mask, center)

        # Replace the blended region in the panorama
        panorama[overlap_start:overlap_end, :] = blended
        
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
