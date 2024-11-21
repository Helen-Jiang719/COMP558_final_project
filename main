import os
from PIL import Image
import numpy as np
import cv2

# Set the paths
input_folder = 'data/raw/'           # Folder containing the original images
output_folder = 'data/processed/'     # Folder to store the processed images
target_size = (800, 600)              # Target size for resizing (width, height)

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)


def preprocess_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        # Process only image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')): #TODO: change later
            try:
                # Open the image
                img_path = os.path.join(input_folder, filename)
                with Image.open(img_path) as img:
                    # Resize the image
                    img = img.resize(target_size, Image.ANTIALIAS)
                    
                    # Save the processed image in the output folder
                    processed_path = os.path.join(output_folder, filename)
                    img.save(processed_path)
                    print(f"Processed and saved: {processed_path}")
            
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

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
    keypoints_list = []
    descriptors_list = []
    
    # Step 1: Detect keypoints for each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png')):  # Filter images
            try:
                # Read the image
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)

                # Convert image to grayscale
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect keypoints and compute descriptors
                keypoints, descriptors = sift.detectAndCompute(img_gray, None)
                
                # Store images, keypoints, and descriptors for later matching
                images.append(img)
                keypoints_list.append(keypoints)
                descriptors_list.append(descriptors)

                # Draw keypoints
                img_kp = cv2.drawKeypoints(img_gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # Save out image with keypoints
                output_path = os.path.join(output_folder, f"kp_{filename}")
                cv2.imwrite(output_path, img_kp)

                print(f"Keypoints detected and saved for {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Step 2: Perform FLANN-based matching between all image pairs
    def flann_feature_matching(img1, img2, kp1, kp2, des1, des2, img1_name, img2_name):
        # Convert descriptors to np.float32
        des1 = np.asarray(des1, dtype=np.float32)
        des2 = np.asarray(des2, dtype=np.float32)

        # FLANN-based feature matcher
        index_params = dict(algorithm=1, table_number=5, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

        # Initialize FLANN-based matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors
        if des1 is not None and len(des1) > 2 and des2 is not None and len(des2) > 2:
            matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply the ratio test to filter out weak matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Draw matches on the image
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Save the visualized image with matches
        output_path = os.path.join(output_folder, f"matches_{img1_name}_{img2_name}.jpg")
        cv2.imwrite(output_path, img_matches)
        print(f"Matches visualized and saved for {img1_name} and {img2_name}")
    
    # Step 3: Compare each image with every other image (all-pair comparison)
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            img1 = images[i]
            img2 = images[j]
            kp1 = keypoints_list[i]
            kp2 = keypoints_list[j]
            des1 = descriptors_list[i]
            des2 = descriptors_list[j]
            
            img1_name = os.path.basename(os.listdir(input_folder)[i])
            img2_name = os.path.basename(os.listdir(input_folder)[j])

            # Perform FLANN matching
            flann_feature_matching(img1, img2, kp1, kp2, des1, des2, img1_name, img2_name)

# Run the SIFT keypoint detection and FLANN feature matching
keypoint_detector_sift(resized_ims, detected_ims)
