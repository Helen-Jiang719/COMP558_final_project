import cv2
import numpy as np
import os

class RANSACMatcher:
    """
    The RANSACMatcher class performs feature matching between image pairs
    and uses the RANSAC algorithm to compute a robust homography matrix.
    This matrix represents the transformation needed to align one image with another.
    """
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def match_and_estimate(self, images, keypoints_list, descriptors_list, filenames):
        homographies = []  # Init array to tore homography matrices
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                try:
                    img1, img2 = images[i], images[j] # Get the images
                    kp1, kp2 = keypoints_list[i], keypoints_list[j] # Get the keypoints
                    des1, des2 = descriptors_list[i], descriptors_list[j] # Get the descriptors
                    # Brute-force matcher
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) # Normalise the descriptors using L2 and dont use cross check so we get more matches
                    matches = bf.knnMatch(des1, des2, k=2) # Match the descriptors using k=2 for k nearest neighbours
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance] # Get the good matches using Lowe's ratio test with a ratio of 0.7 
                    print(f"Good matches between {filenames[i]} and {filenames[j]}: {len(good_matches)}")

                    if len(good_matches) >= 4: # At least 4 matches are needed to estimate a homography
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # Get the source points
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # Get the destination points
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # Estimate the homography matrix using RANSAC with a maximum of 5.0 pixels error
                        matches_mask = mask.ravel().tolist() # Get the mask
                        print(f"Homography matrix for {filenames[i]} and {filenames[j]}:\n{H}")
                        homographies.append((H, i, j))  # Save the homography and indices
                        # Visualise matches after RANSAC
                        draw_params = dict(matchColor=(0, 255, 0),
                                           singlePointColor=(255, 0, 0),
                                           matchesMask=matches_mask,
                                           flags=cv2.DrawMatchesFlags_DEFAULT)
                        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
                        output_path = os.path.join(self.output_folder, f"ransac_matches_{filenames[i]}_{filenames[j]}.jpg")
                        cv2.imwrite(output_path, img_matches)
                        print(f"RANSAC matches visualized and saved for {filenames[i]} and {filenames[j]}")
                    else:
                        print(f"Not enough matches between {filenames[i]} and {filenames[j]} for RANSAC.")
                except Exception as e:
                    print(f"Error matching {filenames[i]} and {filenames[j]}: {e}")
        return homographies
