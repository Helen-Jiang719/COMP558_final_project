from preprocessing import Preprocessor
from SIFT import KeypointDetector
from RANSAC import RANSACMatcher
from homographyWarp import HomographyWarper
from imageStitch import ImageStitcher

input_folder = 'data/raw/flower'
processed_folder = 'data/processed/flowerProcessed'
keypoints_folder = 'data/keypointsFlower'
ransac_folder = 'data/keypoints/ransac_matchesFlower'
warp_folder = 'data/keypoints/warped_imagesFlower'
stitched_output = 'data/results/stitched_panoramaFlower.jpg'

# Preprocess images
preprocessor = Preprocessor(input_folder, processed_folder)
preprocessor.preprocess_images()

# Detect keypoints using SIFT
detector = KeypointDetector(processed_folder, keypoints_folder)
images, keypoints_list, descriptors_list, filenames = detector.detectKeypoints()

# Perform RANSAC matching to estimate homographies
matcher = RANSACMatcher(ransac_folder)
homographies = matcher.match_and_estimate(images, keypoints_list, descriptors_list, filenames)

# Warp images using the homographies for each pair of images
warper = HomographyWarper(warp_folder)
warped_images = []
for H, i, j in homographies:
    warped_images.append(warper.warp_images(images[i], images[j], H, filenames[i], filenames[j]))


