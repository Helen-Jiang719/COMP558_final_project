import cv2
import os

class KeypointDetector:
    """
    The KeypointDetector class is responsible for identifying keypoints and descriptors
    in images using the SIFT.
    """
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.sift = cv2.SIFT_create()

    def detectKeypoints(self):
        keypoints_list = []
        descriptors_list = []
        images = []
        filenames = []
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    img_path = os.path.join(self.input_folder, filename)
                    img = cv2.imread(img_path)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
                    keypoints, descriptors = self.sift.detectAndCompute(img_gray, None) # Detect keypoints and descriptors
                    images.append(img) # Append the image to the list
                    keypoints_list.append(keypoints) # Append the keypoints to the list
                    descriptors_list.append(descriptors) # Append the descriptors to the list
                    filenames.append(filename) # Append the filename to the list
                    # Visualise and save keypoints
                    img_kp = cv2.drawKeypoints(img_gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw the keypoints on the image using CV2
                    output_path = os.path.join(self.output_folder, f"kp_{filename}")
                    cv2.imwrite(output_path, img_kp)
                    print(f"Keypoints detected and saved for {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return images, keypoints_list, descriptors_list, filenames
