import cv2
import os

class HomographyWarper:
    """
    The HomographyWarper class is responsible for applying the homography matrix
    to an image to align it with another image. This is a crucial step in creating
    a panorama, where one image is transformed (warped) to fit into the coordinate
    system of another image.
    """
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def warp_images(self, img1, img2, H, filename1, filename2):
        try:
            height, width, _ = img2.shape # Get the dimensions of the second image
            warped_image = cv2.warpPerspective(img1, H, (width, height)) # Warp the first image using the homography matrix
            warp_output_path = os.path.join(self.output_folder, f"warped_{filename1}_{filename2}.jpg")
            cv2.imwrite(warp_output_path, warped_image)
            print(f"Warped image saved: {warp_output_path}")
        except Exception as e:
            print(f"Error warping images {filename1} and {filename2}: {e}")
