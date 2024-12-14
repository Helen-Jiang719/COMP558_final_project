import cv2
import numpy as np
def stitch(images):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise Exception(f"Stitching failed with status code {status}")
    return panorama
def read_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image")
        images.append(img)
    return images
if __name__ == "__main__":

    image_paths = [
   "data/raw/NSH/medium01.jpg",  
    "data/raw/NSH/medium02.jpg",
]
    try:
        images = read_images(image_paths)
        panorama = stitch(images)
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("MultiBandPanorama.jpg", panorama)
        print("Panorama saved as vertical_panorama.jpg")
    except Exception as e:
        print(f"Error: {e}")