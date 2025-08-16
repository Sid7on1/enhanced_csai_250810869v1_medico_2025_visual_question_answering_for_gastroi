import logging
import os
import shutil
import tempfile
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Image preprocessing utilities for computer vision tasks.

    ...

    Attributes
    ----------
    input_folder: str
        Path to the folder containing input images.
    output_folder: str
        Path to the folder where preprocessed images will be saved.
    temp_folder: str
        Path to a temporary folder for intermediate files.
    image_size: Tuple[int, int]
        Target size for resizing images.
    mean: Tuple[float, float, float]
        Mean values for image normalization.
    std: Tuple[float, float, float]
        Standard deviation values for image normalization.
    random_seed: int
        Random seed for reproducibility.

    Methods
    -------
    preprocess_images(self, image_paths: List[str]) -> List[str]:
        Preprocesses a list of images and saves the results.

    preprocess_image(self, image_path: str) -> np.array:
        Preprocesses a single image and returns the processed array.

    normalize_image(self, image: np.array) -> np.array:
        Normalizes an image based on the mean and standard deviation.

    resize_image(self, image: np.array, size: Tuple[int, int]) -> np.array:
        Resizes an image to the specified size.

    create_folders(self) -> None:
        Creates the necessary input, output, and temp folders.

    delete_temp_folder(self) -> None:
        Deletes the temporary folder and its contents.
    """

    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        temp_folder: str,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        random_seed: int = 42,
    ):
        """
        Initializes the ImagePreprocessor with the given parameters.

        Parameters
        ----------
        input_folder: str
            Path to the folder containing input images.
        output_folder: str
            Path to the folder where preprocessed images will be saved.
        temp_folder: str
            Path to a temporary folder for intermediate files.
        image_size: Tuple[int, int], optional
            Target size for resizing images (default is (224, 224)).
        mean: Tuple[float, float, float], optional
            Mean values for image normalization (default are ImageNet values).
        std: Tuple[float, float, float], optional
            Standard deviation values for image normalization (default are ImageNet values).
        random_seed: int, optional
            Random seed for reproducibility (default is 42).
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.temp_folder = temp_folder
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.random_seed = random_seed

        self.create_folders()

    def preprocess_images(self, image_paths: List[str]) -> List[str]:
        """
        Preprocesses a list of images and saves the results.

        Parameters
        ----------
        image_paths: List[str]
            List of paths to the input images.

        Returns
        -------
        List[str]
            List of paths to the preprocessed images.
        """
        preprocessed_images = []
        for image_path in image_paths:
            preprocessed_image = self.preprocess_image(image_path)
            output_path = os.path.join(self.output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, preprocessed_image)
            preprocessed_images.append(output_path)
        return preprocessed_images

    def preprocess_image(self, image_path: str) -> np.array:
        """
        Preprocesses a single image and returns the processed array.

        Parameters
        ----------
        image_path: str
            Path to the input image.

        Returns
        -------
        np.array
            Preprocessed image as a numpy array.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            image = self.resize_image(image, self.image_size)
            image = self.normalize_image(image)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def normalize_image(self, image: np.array) -> np.array:
        """
        Normalizes an image based on the mean and standard deviation.

        Parameters
        ----------
        image: np.array
            Input image to be normalized.

        Returns
        -------
        np.array
            Normalized image.
        """
        image = image.astype(np.float32)
        image /= 255.0
        image -= np.array(self.mean)
        image /= np.array(self.std)
        return image

    def resize_image(self, image: np.array, size: Tuple[int, int]) -> np.array:
        """
        Resizes an image to the specified size.

        Parameters
        ----------
        image: np.array
            Input image to be resized.
        size: Tuple[int, int]
            Target size for the image.

        Returns
        -------
        np.array
            Resized image.
        """
        return cv2.resize(image, size)

    def create_folders(self) -> None:
        """
        Creates the necessary input, output, and temp folders.

        Raises
        ------
        ValueError
            If any of the required folders already exist.
        """
        if os.path.exists(self.input_folder):
            raise ValueError(f"Input folder already exists: {self.input_folder}")
        os.makedirs(self.input_folder)

        if os.path.exists(self.output_folder):
            raise ValueError(f"Output folder already exists: {self.output_folder}")
        os.makedirs(self.output_folder)

        if os.path.exists(self.temp_folder):
            raise ValueError(f"Temp folder already exists: {self.temp_folder}")
        os.makedirs(self.temp_folder)

    def delete_temp_folder(self) -> None:
        """
        Deletes the temporary folder and its contents.
        """
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)

class ImageCrops:
    """
    Utilities for cropping images based on specific criteria.

    ...

    Attributes
    ----------
    input_folder: str
        Path to the folder containing input images.
    output_folder: str
        Path to the folder where cropped images will be saved.
    crop_size: Tuple[int, int]
        Size of the crops to extract.
    num_crops: int
        Number of crops to extract from each image.
    random_seed: int
        Random seed for reproducibility.

    Methods
    -------
    extract_crops(self, image_paths: List[str]) -> List[str]:
        Extracts crops from a list of images and saves the results.

    extract_crop(self, image: np.array) -> List[np.array]:
        Extracts a single crop from an image and returns the array.

    create_folders(self) -> None:
        Creates the necessary input and output folders.

    delete_output_folder(self) -> None:
        Deletes the output folder and its contents.
    """

    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        crop_size: Tuple[int, int] = (224, 224),
        num_crops: int = 10,
        random_seed: int = 42,
    ):
        """
        Initializes the ImageCrops with the given parameters.

        Parameters
        ----------
        input_folder: str
            Path to the folder containing input images.
        output_folder: str
            Path to the folder where cropped images will be saved.
        crop_size: Tuple[int, int], optional
            Size of the crops to extract (default is (224, 224)).
        num_crops: int, optional
            Number of crops to extract from each image (default is 10).
        random_seed: int, optional
            Random seed for reproducibility (default is 42).
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.random_seed = random_seed

        self.create_folders()

    def extract_crops(self, image_paths: List[str]) -> List[str]:
        """
        Extracts crops from a list of images and saves the results.

        Parameters
        ----------
        image_paths: List[str]
            List of paths to the input images.

        Returns
        -------
        List[str]
            List of paths to the cropped images.
        """
        cropped_images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            crops = self.extract_crop(image)
            for i, crop in enumerate(crops):
                output_path = os.path.join(self.output_folder, f"{os.path.basename(image_path)}_{i}.jpg")
                cv2.imwrite(output_path, crop)
                cropped_images.append(output_path)
        return cropped_images

    def extract_crop(self, image: np.array) -> List[np.array]:
        """
        Extracts a single crop from an image and returns the array.

        Parameters
        ----------
        image: np.array
            Input image from which to extract the crop.

        Returns
        -------
        List[np.array]
            List containing a single crop from the image.
        """
        height, width, _ = image.shape
        start_x = np.random.randint(0, width - self.crop_size[0])
        start_y = np.random.randint(0, height - self.crop_size[1])
        end_x = start_x + self.crop_size[0]
        end_y = start_y + self.crop_size[1]
        crop = image[start_y:end_y, start_x:end_x]
        return [crop]

    def create_folders(self) -> None:
        """
        Creates the necessary input and output folders.

        Raises
        ------
        ValueError
            If any of the required folders already exist.
        """
        if os.path.exists(self.input_folder):
            raise ValueError(f"Input folder already exists: {self.input_folder}")
        os.makedirs(self.input_folder)

        if os.path.exists(self.output_folder):
            raise ValueError(f"Output folder already exists: {self.output_folder}")
        os.makedirs(self.output_folder)

    def delete_output_folder(self) -> None:
        """
        Deletes the output folder and its contents.
        """
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

def load_images(folder: str) -> List[str]:
    """
    Loads a list of image paths from a folder.

    Parameters
    ----------
    folder: str
        Path to the folder containing images.

    Returns
    -------
    List[str]
        List of paths to the images in the folder.
    """
    image_paths = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            image_paths.append(os.path.join(folder, filename))
    return image_paths

def save_image_paths(image_paths: List[str], filename: str) -> None:
    """
    Saves a list of image paths to a text file.

    Parameters
    ----------
    image_paths: List[str]
        List of paths to the images.
    filename: str
        Name of the file to save the paths to.
    """
    with open(filename, "w") as f:
        for path in image_paths:
            f.write(f"{path}\n")

def load_image_paths(filename: str) -> List[str]:
    """
    Loads a list of image paths from a text file.

    Parameters
    ----------
    filename: str
        Name of the file containing the image paths.

    Returns
    -------
    List[str]
        List of paths to the images.
    """
    with open(filename, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    return image_paths

# Example usage
if __name__ == "__main__":
    input_folder = "input_images"
    output_folder = "preprocessed_images"
    temp_folder = "temp_files"

    preprocessor = ImagePreprocessor(input_folder, output_folder, temp_folder)
    image_paths = load_images(input_folder)

    preprocessed_images = preprocessor.preprocess_images(image_paths)
    logger.info(f"Preprocessed images saved to: {output_folder}")

    # Clean up temp folder
    preprocessor.delete_temp_folder()

    # Example of cropping images
    cropper = ImageCrops(output_folder, "cropped_images")
    cropped_images = cropper.extract_crops(preprocessed_images)
    logger.info(f"Cropped images saved to: cropped_images")

    # Clean up output folder
    cropper.delete_output_folder()