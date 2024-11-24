import os
from PIL import Image
import numpy as np
import cv2
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Set the paths
input_folder = 'data/raw/'           # Folder containing the original images
output_folder = 'data/processed/'     # Folder to store the processed images
keypoints_folder = 'data/keypoints/' # Folder for keypoints visualization
matches_folder = os.path.join(keypoints_folder, "matches/")  # Matches visualization
os.makedirs(output_folder, exist_ok=True)
os.makedirs(keypoints_folder, exist_ok=True)
os.makedirs(matches_folder, exist_ok=True)
target_size = (800, 600)              # Target size for resizing (width, height)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def preprocess_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        # Process only image files
        if filename.lower().endswith((IMAGE_EXTENSIONS)): #TODO: change later
            try:
                # Open the image
                img_path = os.path.join(input_folder, filename)
                with Image.open(img_path) as img:
                    # Resize the image
                    img = img.resize(target_size, Image.ANTIALIAS)
                    # Save the processed image in the output folder
                    processed_path = os.path.join(output_folder, filename)
                    img.save(processed_path)
                    logging.info(f"Processed and saved: {processed_path}")
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")

# Run the preprocessing function
preprocess_images(input_folder, output_folder)


# Implementation for SIFT keypoint detection
# Directories for input and output images
resized_ims = 'data/processed/'
detected_ims = 'data/keypoints/'
os.makedirs(detected_ims, exist_ok=True)

def keypoint_detector_sift(input_folder, output_folder):
    # Create a sift object
    sift = cv2.SIFT_create()
    images = []
    filenames = []
    keypoints_list = []
    descriptors_list = []
    
    # Step 1: Detect keypoints for each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png')):  # Filter images
            try:
                # Read the image
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
                keypoints, descriptors = sift.detectAndCompute(img_gray, None)                 # Detect keypoints and compute descriptors
                
                # Store images, keypoints, and descriptors for later matching
                images.append(img)
                filenames.append(filename)
                keypoints_list.append(keypoints)
                descriptors_list.append(descriptors)

                # Draw keypoints
                img_kp = cv2.drawKeypoints(img_gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # Save out image with keypoints
                output_path = os.path.join(output_folder, f"kp_{filename}")
                cv2.imwrite(output_path, img_kp)
                logging.info(f"Keypoints detected and saved for {filename}")

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")

    # Step 2: Perform FLANN-based matching and homography estimation
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            img1, img2 = images[i], images[j]
            kp1, kp2 = keypoints_list[i], keypoints_list[j]
            des1, des2 = descriptors_list[i], descriptors_list[j]

            try:
                # Convert descriptors to np.float32 for FLANN
                des1 = np.asarray(des1, dtype=np.float32)
                des2 = np.asarray(des2, dtype=np.float32)

                # FLANN matcher
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                matches = flann.knnMatch(des1, des2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                logging.info(f"{len(good_matches)} good matches found between {filenames[i]} and {filenames[j]}")

                # Draw matches
                img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                match_output_path = os.path.join(matches_folder, f"matches_{filenames[i]}_{filenames[j]}.jpg")
                cv2.imwrite(match_output_path, img_matches)

                # Estimate homography
                if len(good_matches) > 4:  # Minimum 4 points required for homography
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    logging.info(f"Homography estimated between {filenames[i]} and {filenames[j]}")

                    # Warp the first image onto the second
                    height, width, _ = img2.shape
                    warped_image = cv2.warpPerspective(img1, H, (width, height))
                    warp_output_path = os.path.join(matches_folder, f"warped_{filenames[i]}_{filenames[j]}.jpg")
                    cv2.imwrite(warp_output_path, warped_image)
                    logging.info(f"Warped image saved: {warp_output_path}")

            except Exception as e:
                logging.error(f"Error matching or estimating homography between {filenames[i]} and {filenames[j]}: {e}")

# Run the SIFT keypoint detection and FLANN feature matching
keypoint_detector_sift(resized_ims, detected_ims)
