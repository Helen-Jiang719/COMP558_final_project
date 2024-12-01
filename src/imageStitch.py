import cv2
import numpy as np

class ImageStitcher:
    def __init__(self, canvas_size=(2000, 1000)):
        self.canvas_size = canvas_size

    def warp_image(self, image, homography):
        warped = cv2.warpPerspective(image, homography, self.canvas_size)
        return warped
    def stitch_images(self, images, homographies):
        """
        Stitch multiple images into a panorama using the provided homographies.
        """
        if len(image.shape) == 2:  # If the image is grayscale
         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        print("Image shapes:")
        for i, img in enumerate(images):
            print(f"Image {i} shape: {img.shape}")  

        for i in range(len(images)):
            if images[i].shape[:2] != images[0].shape[:2]:
                 images[i] = cv2.resize(images[i], (images[0].shape[1], images[0].shape[0]))


        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) # Create a large canvas for the panorama to be stitched onto
         # Warp the first image directly onto the canvas by placing it in the top left corner
        canvas[:images[0].shape[0], :images[0].shape[1], :] = images[0]  # Explicitly use slicing for 3 channels 
        # Warp and blend remaining images
        for i, (image, H) in enumerate(zip(images[1:], homographies)):
            warped = self.warp_image(image, H)
            # Blend the warped image with the canvas
            mask = (warped > 0).astype(np.uint8)  # Mask for non-zero pixels
            canvas = cv2.addWeighted(canvas, 0.5, warped, 0.5, 0) 
            print(f"Image {i + 1} stitched.")

        return canvas
