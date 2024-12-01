from PIL import Image
import os

class Preprocessor:
    """
    The Preprocessor class is responsible for preparing raw images for further processing.
    It resizes images to a standard size so that they are consistent and easier to handle
    during keypoint detection and stitching.
    """
    # Initialising the Preprocessor with the input and output folder paths and the target size for the images
    def __init__(self, input_folder, output_folder, target_size=(800, 600)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_size = target_size
        os.makedirs(self.output_folder, exist_ok=True) # Create the output folder if it doesnt exist

    def preprocess_images(self):
        """
        Resize all images in the input folder to the target size and save them
        to the output folder. This step ensures all images are uniform, which
        simplifies later steps in the pipeline.
        """
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    img_path = os.path.join(self.input_folder, filename)
                    with Image.open(img_path) as img: 
                        img = img.resize(self.target_size, Image.Resampling.LANCZOS) # Resize the image using the LANCZOS filter
                        processed_path = os.path.join(self.output_folder, filename)
                        if not os.path.isfile(processed_path): # If the file doesnt exist, save it
                            img.save(processed_path)
                            print(f"Processed and saved: {processed_path}")
                except Exception as e:
                    print(f"Failed to process {filename}: {e}") 
