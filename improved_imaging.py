# Required imports for the M&M Counter System
import cv2 # computer vision library used for image processing and other related features to machine vision
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk # used for image reformatting and resizing -> just to make the image mutble for our code
import json
import datetime
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import Counter
import time
import sys
import traceback

# EDITS THAT MUST BE MADE
# - generalize circle detection by replacing with ellipses to allow for more variablility without loss of accuracy
# - improve color detection, specifically for brown and orange ranges
# - improve color detection under varying light conditions (e.g. light from different angles, different light levels - more or less shine and brighter or darker colours)
# - try to limit assumptions; can't be sure the predefined colour set will always be the different colours being detected


# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WebcamCapture: # all this does is take a picture so it may be passed to te class ImageProcessor
    """
    Handles webcam operations including initialization, capture, and resource management.
    """
    
    def __init__(self, camera_id: int = 0):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize attributes
        self.camera = None
        self.is_capturing = False
        self.camera_id = camera_id
        
        # Default camera settings
        self.frame_width = 1280
        self.frame_height = 720
        self.fps = 30
        
        # Camera parameters
        self.camera_settings = { # configuration
            cv2.CAP_PROP_FRAME_WIDTH: self.frame_width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.frame_height,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_AUTOFOCUS: 1,
            cv2.CAP_PROP_BRIGHTNESS: 128,
            cv2.CAP_PROP_CONTRAST: 128,
            cv2.CAP_PROP_SATURATION: 128
        }
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = None
        
    def initialize_camera(self, camera_id: Optional[int] = None) -> bool:
        try:
            if camera_id is not None:
                self.camera_id = camera_id
                
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            for prop, value in self.camera_settings.items():
                self.camera.set(prop, value)
                
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height}")
            
            self.frame_count = 0
            self.start_time = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            return False
            
    def capture_frame(self) -> Optional[np.ndarray]:
        if self.camera is None:
            self.logger.error("Camera not initialized")
            return None
            
        try:
            ret, frame = self.camera.read()
            
            if ret:
                self.frame_count += 1
                return frame
            else:
                self.logger.error("Failed to capture frame")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {str(e)}")
            return None
            
    def get_fps(self) -> float:
        if self.start_time is None or self.frame_count == 0:
            return 0.0
            
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        
    def get_camera_properties(self) -> Dict:
        if self.camera is None:
            return {}
            
        properties = {
            'width': self.camera.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.camera.get(cv2.CAP_PROP_FPS),
            'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
            'auto_focus': self.camera.get(cv2.CAP_PROP_AUTOFOCUS),
            'actual_fps': self.get_fps()
        }
        return properties
        
    def set_camera_property(self, property_id: int, value: float) -> bool:
        if self.camera is None:
            return False
            
        try:
            return self.camera.set(property_id, value)
        except Exception as e:
            self.logger.error(f"Error setting camera property: {str(e)}")
            return False
            
    def set_resolution(self, width: int, height: int) -> bool:
        try:
            success = True
            success &= self.set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, width)
            success &= self.set_camera_property(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if success:
                self.frame_width = width
                self.frame_height = height
                self.logger.info(f"Resolution set to {width}x{height}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            return False
            
    def start_preview(self, window_name: str = "Camera Preview") -> None:
        if self.camera is None:
            self.logger.error("Camera not initialized")
            return
            
        try:
            self.is_capturing = True
            
            while self.is_capturing:
                frame = self.capture_frame()
                
                if frame is not None:
                    cv2.imshow(window_name, frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            cv2.destroyWindow(window_name)
            
        except Exception as e:
            self.logger.error(f"Error in preview: {str(e)}")
            self.is_capturing = False
            
    def stop_preview(self) -> None:
        self.is_capturing = False
        
    def release(self) -> None:
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                self.is_capturing = False
                self.logger.info("Camera resources released")
                
        except Exception as e:
            self.logger.error(f"Error releasing camera: {str(e)}")
            
    def __del__(self):
        self.release()

class ImageProcessor: # processes the image to be used for M&M detection and differentiation
    """
    Handles image processing operations for M&M detection.
    
    Attributes:
        logger: Logging instance for error tracking
        blur_kernel_size: Size of Gaussian blur kernel
        adaptive_block_size: Block size for adaptive thresholding
    """
    
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Processing parameters
        self.blur_kernel_size = (7, 7)
        self.adaptive_block_size = 11
        self.clahe_parameters = {
            'clip_limit': 3.0,
            'tile_grid_size': (8, 8)
        }
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess image for M&M detection.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple containing:
                - Grayscale processed image
                - HSV image
                - Blurred image
                
        Raises:
            ValueError: If input image is None or empty
            Exception: For other processing errors
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
                
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, self.blur_kernel_size, 0)
            
            # Convert blurred image to grayscale
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            
            return gray, hsv, blurred
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def preprocess_for_ellipses(self, gray: np.ndarray) -> np.ndarray:
        """
        Preprocess grayscale image for better ellipse detection.
        Args:
            gray: Input grayscale image.

        Returns:
            Preprocessed binary image suitable for contour extraction.
        """
        try:
            # Adaptive thresholding for uneven lighting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Morphological operations to clean up edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            return binary
        except Exception as e:
            self.logger.error(f"Error in preprocessing for ellipses: {e}")
            raise
            
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better detection.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Create a copy of the input image
            enhanced = image.copy()
            
            # Convert to LAB color space
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            
            # Split the LAB image into L, A, and B channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_parameters['clip_limit'],
                tileGridSize=self.clahe_parameters['tile_grid_size']
            )
            cl = clahe.apply(l)
            
            # Merge the CLAHE enhanced L-channel back with A and B channels
            limg = cv2.merge((cl, a, b))
            
            # Convert back to BGR color space
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error in image enhancement: {str(e)}")
            raise
            
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                 alpha: float = 1.0, 
                                 beta: int = 0) -> np.ndarray:
        """
        Adjust image brightness and contrast.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Adjusted image
        """
        try:
            # Validate parameters
            alpha = max(0.0, min(3.0, alpha))
            beta = max(0, min(100, beta))
            
            # Apply brightness and contrast adjustment
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            return adjusted
            
        except Exception as e:
            self.logger.error(f"Error in brightness/contrast adjustment: {str(e)}")
            raise
            
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to the image.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        try:
            # Apply non-local means denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
            
        except Exception as e:
            self.logger.error(f"Error in denoising: {str(e)}")
            raise
            
    def process_for_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete processing pipeline for M&M detection.
        
        Args:
            image: Input image
            
        Returns:
            Tuple containing processed images for detection:
                - Grayscale image
                - HSV image
                - Blurred image
        """
        try:
            # Enhance image
            enhanced = self.enhance_image(image)
            
            # Denoise
            denoised = self.denoise_image(enhanced)
            
            # Adjust brightness and contrast
            adjusted = self.adjust_brightness_contrast(denoised, 1.2, 10)
            
            # Get processed images
            gray, hsv, blurred = self.preprocess_image(adjusted)
            
            return gray, hsv, blurred
            
        except Exception as e:
            self.logger.error(f"Error in detection processing: {str(e)}")
            raise
            
    def get_processing_parameters(self) -> dict:
        """
        Get current processing parameters.
        
        Returns:
            Dictionary of current processing parameters
        """
        return {
            'blur_kernel_size': self.blur_kernel_size,
            'adaptive_block_size': self.adaptive_block_size,
            'clahe_parameters': self.clahe_parameters
        }
        
    def set_processing_parameters(self, 
                                blur_kernel_size: Optional[Tuple[int, int]] = None,
                                adaptive_block_size: Optional[int] = None,
                                clahe_clip_limit: Optional[float] = None,
                                clahe_grid_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Set processing parameters.
        
        Args:
            blur_kernel_size: Size of Gaussian blur kernel
            adaptive_block_size: Block size for adaptive thresholding
            clahe_clip_limit: Clip limit for CLAHE
            clahe_grid_size: Grid size for CLAHE
        """
        try:
            if blur_kernel_size is not None:
                self.blur_kernel_size = blur_kernel_size
            if adaptive_block_size is not None:
                self.adaptive_block_size = adaptive_block_size
            if clahe_clip_limit is not None:
                self.clahe_parameters['clip_limit'] = clahe_clip_limit
            if clahe_grid_size is not None:
                self.clahe_parameters['tile_grid_size'] = clahe_grid_size
                
        except Exception as e:
            self.logger.error(f"Error setting processing parameters: {str(e)}")
            raise

class MMDetector:
    '''
    Handles M&M detection and color classification.
    
    Attributes:
        logger: Logging instance for error tracking
        min_radius: Minimum radius for M&M detection
        max_radius: Maximum radius for M&M detection
        min_dist: Minimum distance between detected circles
        param1: First parameter for Hough Circle detection
        param2: Second parameter for Hough Circle detection
        color_ranges: HSV color ranges for M&M classification
    '''
    
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Create an instance of ImageProcessor
        self.image_processor = ImageProcessor()
        
        # Circle detection parameters - Optimized
        self.min_radius = 8
        self.max_radius = 32

        # Ellipse detection parameters
        self.min_radius = 16.8
        self.max_radius = 22.9
        self.min_ratio = 0.5
        self.max_ratio = 1.5

        self.min_dist = 20 # Minimum distance (in pixels) between M&Ms
        self.param1 = 30  # Edge detection parameter - lower means more edges detected
        self.param2 = 30  # Circle detection parameter - how confident the model must be that it is in fact a circle to detect it
        
        # Improved HSV color ranges
        self.color_ranges = {
            'red': {
                'lower1': np.array([0, 120, 80]),  # Higher S and V to avoid orange
                'upper1': np.array([10, 255, 255]),  # Narrowed upper bound for red
                'lower2': np.array([160, 120, 80]),  # Wrap-around reds
                'upper2': np.array([179, 255, 255])
            },
            'blue': {
                'lower': np.array([96, 80, 50]),  # Slightly lower V for dim lighting
                'upper': np.array([130, 255, 255])
            },
            'green': {
                'lower': np.array([47, 80, 50]),  # Robust for dim light
                'upper': np.array([80, 255, 219])
            },
            'yellow': {
                'lower': np.array([20, 60, 100]),  # Includes muted yellow shades
                'upper': np.array([50, 255, 255])  # Extended H to cover yellow-green boundary
            },
            'orange': {
                'lower': np.array([11, 120, 80]),  # Includes orange leaning toward red
                'upper': np.array([20, 255, 255])  # Ensures distinctness from yellow
            },
            'brown': {
                'lower': np.array([5, 40, 20]),  # Expanded lower S and V for lighting variations
                'upper': np.array([20, 200, 120])  # Higher upper V for shiny M&Ms
            }
        }
        # Detection parameters history for analysis
        self.detection_history = []

    def set_detection_params(self, min_radius=None, max_radius=None, min_ratio=None, max_ratio=None):
        """
        Allows dynamic updating of ellipse detection parameters.
        """
        if min_radius is not None:
            self.min_radius = max(0, min_radius)
        if max_radius is not None:
            self.max_radius = max(self.min_radius, max_radius)
        if min_ratio is not None:
            self.min_ratio = max(0, min_ratio)
        if max_ratio is not None:
            self.max_ratio = max(self.min_ratio, max_ratio)

        self.logger.info(f"Updated ellipse parameters: min_radius={self.min_radius}, max_radius={self.max_radius}, "
                         f"min_ratio={self.min_ratio}, max_ratio={self.max_ratio}")


    def detect_circles(self, gray: np.ndarray) -> np.ndarray:
        '''
        Detect circles in the grayscale image using Hough Circle Transform.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Array of detected circles (x, y, radius)
        '''
        try:
            # Validate input
            if gray is None or gray.size == 0:
                raise ValueError("Invalid input image")
                
            # Apply Hough Circle Transform
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=self.min_dist,
                param1=self.param1,
                param2=self.param2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )
            
            # Process detected circles
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                # Record detection parameters and results
                self.detection_history.append({
                    'parameters': self.get_parameters(),
                    'circles_detected': len(circles[0])
                })
                
                return circles[0]
            return np.array([])
            
        except Exception as e:
            self.logger.error(f"Error in circle detection: {str(e)}")
            raise

    def detect_ellipses(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect and refine ellipses in the grayscale image.

        Args:
            gray: Grayscale input image.

        Returns:
            Array of refined ellipses (x, y, major_axis, minor_axis, angle).
        """
        try:
            if gray is None or gray.size == 0:
                raise ValueError("Invalid input image")

            # Preprocess for contours
            binary = self.image_processor.preprocess_for_ellipses(gray)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours
            valid_contours = self.filter_contours(contours)

            ellipses = []
            for contour in valid_contours:
                if len(contour) >= 5:  # Minimum points required for ellipse fitting
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (major_axis, minor_axis), angle = ellipse

                    # Filter by size and aspect ratio with relaxed criteria
                    if (
                        self.min_radius - 2 <= major_axis / 2 <= self.max_radius + 2 and
                        self.min_radius - 2 <= minor_axis / 2 <= self.max_radius + 2 and
                        self.min_ratio - 0.1 <= (minor_axis / major_axis) <= self.max_ratio + 0.1
                    ):
                        ellipses.append((int(x), int(y), int(major_axis), int(minor_axis), angle))

            ellipses = np.array(ellipses)

            # Apply nested ellipse filtering
            ellipses = self.filter_nested_ellipses(ellipses)

            # Apply non-maximum suppression to remove overlaps
            ellipses = self.non_max_suppression(ellipses)

            self.logger.info(f"Detected {len(ellipses)} refined ellipses.")
            return ellipses

        except Exception as e:
            self.logger.error(f"Error detecting ellipses: {e}")
            raise


    def adjust_color_ranges(self, sample_hsv: np.ndarray) -> None:
        """
        Dynamically adjust color ranges based on a sample HSV image.
        Args:
            sample_hsv: Sample HSV image for color calibration.
        """
        try:
            # Use clustering to detect dominant colors and set ranges
            reshaped_hsv = sample_hsv.reshape((-1, 3))
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=len(self.color_ranges))
            kmeans.fit(reshaped_hsv)
            centers = kmeans.cluster_centers_

            new_ranges = {}
            for idx, (color, center) in enumerate(zip(self.color_ranges.keys(), centers)):
                h, s, v = center
                h_low, h_high = max(h - 10, 0), min(h + 10, 179)
                s_low, s_high = max(s - 40, 0), min(s + 40, 255)
                v_low, v_high = max(v - 40, 0), min(v + 40, 255)
                new_ranges[color] = {'lower': np.array([h_low, s_low, v_low]),
                                    'upper': np.array([h_high, s_high, v_high])}

            self.color_ranges = new_ranges
            self.logger.info("Color ranges adjusted dynamically.")
        except Exception as e:
            self.logger.error(f"Error adjusting color ranges: {e}")

       

    def identify_colors(self, hsv: np.ndarray, circles: np.ndarray) -> List[str]:
        '''
        Identify colors of detected M&Ms.
        
        Args:
            hsv: HSV format image
            circles: Detected circles
            
        Returns:
            List of color names for each detected M&M
        '''
        try:
            if hsv is None or circles is None:
                raise ValueError("Invalid inputs for color identification")
                
            colors = []
            for (x, y, r) in circles:
                # Create mask for current circle
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (x, y), r-5, 255, -1)
                
                # Get mean color in the masked area
                mean_color = cv2.mean(hsv, mask=mask)[:3]
                
                # Identify color
                detected_color = self._classify_color(mean_color)
                colors.append(detected_color)
                
            return colors
            
        except Exception as e:
            self.logger.error(f"Error in color identification: {str(e)}")
            raise
           
    def _classify_color(self, hsv_color: Tuple[float, float, float]) -> str: # why is red a special case??
        '''
        Classify color based on HSV values.
        
        Args:
            hsv_color: HSV color values
            
        Returns:
            Identified color name
        '''
        try:
            hsv_np = np.uint8([[list(hsv_color)]])
            
            # Check each color range
            for color_name, ranges in self.color_ranges.items():
                if color_name == 'red':
                    # Special case for red (wraps around HSV)
                    mask1 = cv2.inRange(hsv_np, ranges['lower1'], ranges['upper1'])
                    mask2 = cv2.inRange(hsv_np, ranges['lower2'], ranges['upper2'])
                    if mask1.any() or mask2.any():
                        return color_name
                else:
                    mask = cv2.inRange(hsv_np, ranges['lower'], ranges['upper'])
                    if mask.any():
                        return color_name
                        
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Error in color classification: {str(e)}")
            raise

 
    def adjust_detection_params(self, min_radius: Optional[int] = None,
                              max_radius: Optional[int] = None,
                              min_dist: Optional[int] = None,
                              param1: Optional[int] = None,
                              param2: Optional[int] = None) -> None: # used to determine the circle parameters for detection
        '''
        Adjust circle detection parameters.
        
        Args:
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            min_dist: Minimum distance between circles
            param1: Edge detection parameter
            param2: Circle detection parameter
        '''
        try:
            if min_radius is not None:
                self.min_radius = max(1, min_radius)
            if max_radius is not None:
                self.max_radius = max(self.min_radius, max_radius)
            if min_dist is not None:
                self.min_dist = max(1, min_dist)
            if param1 is not None:
                self.param1 = max(1, param1)
            if param2 is not None:
                self.param2 = max(1, param2)
                
            self.logger.info("Detection parameters adjusted: " + 
                           f"min_radius={self.min_radius}, " +
                           f"max_radius={self.max_radius}, " +
                           f"min_dist={self.min_dist}, " +
                           f"param1={self.param1}, " +
                           f"param2={self.param2}")
                
        except Exception as e:
            self.logger.error(f"Error adjusting detection parameters: {str(e)}")
            raise
            
    def get_parameters(self) -> Dict: # getter
        '''
        Get current detection parameters.
        
        Returns:
            Current parameter values
        '''
        return {
            'min_radius': self.min_radius,
            'max_radius': self.max_radius,
            'min_dist': self.min_dist,
            'param1': self.param1,
            'param2': self.param2
        }
    
    def process_image(self, gray: np.ndarray, hsv: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Complete processing pipeline for M&M detection and classification.

        Args:
            gray: Grayscale image.
            hsv: HSV format image.

        Returns:
            Tuple containing detected ellipses and their colors.
        """
        try:
            # Detect ellipses
            ellipses = self.detect_ellipses(gray)

            # Identify colors for each detected ellipse
            colors = []
            for x, y, major_axis, minor_axis, angle in ellipses:
                # Create a mask for the ellipse
                mask = np.zeros_like(gray, dtype=np.uint8)
                center = (int(x), int(y))
                axes = (int(major_axis // 2), int(minor_axis // 2))
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
                
                # Compute mean HSV within the mask
                mean_color = cv2.mean(hsv, mask=mask)[:3]
                color = self._classify_color(mean_color)
                colors.append(color)

            return ellipses, colors

        except Exception as e:
            self.logger.error(f"Error in image processing: {e}")
            raise

    
    def draw_results(self, image: np.ndarray, ellipses: np.ndarray, colors: List[str]) -> np.ndarray:
        """
        Draw detection results on the image.

        Args:
            image: Original image.
            ellipses: Detected ellipses.
            colors: Identified colors.

        Returns:
            Image with detection results drawn.
        """
        try:
            result_image = image.copy()

            for (x, y, major_axis, minor_axis, angle), color in zip(ellipses, colors):
                # Draw ellipse outline
                center = (int(x), int(y))
                axes = (int(major_axis // 2), int(minor_axis // 2))
                cv2.ellipse(result_image, center, axes, angle, 0, 360, (0, 255, 0), 2)

                # Draw center point
                cv2.circle(result_image, center, 2, (0, 0, 255), 3)

                # Add color label
                cv2.putText(result_image, color, (int(x - 20), int(y - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            return result_image

        except Exception as e:
            self.logger.error(f"Error drawing results: {e}")
            raise


    def segment_overlapping_objects(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply Watershed segmentation to separate overlapping objects.
        Args:
            gray: Grayscale image.
            mask: Binary mask of objects.
        Returns:
            Segmented image with separated objects.
        """
        try:
            # Distance transform
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(mask, sure_fg)

            # Markers
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            # Apply Watershed
            segmented = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
            return segmented
        except Exception as e:
            self.logger.error(f"Error segmenting objects: {e}")
            raise

    def separate_overlapping_objects(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply distance transform to separate overlapping objects.

        Args:
            binary: Binary mask of objects.

        Returns:
            Modified binary mask with separated objects.
        """
        try:
            # Distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            # Connected components for markers
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[binary == 0] = 0

            # Apply Watershed to separate objects
            markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)
            separated = np.uint8(markers > 1) * 255
            return separated
        except Exception as e:
            self.logger.error(f"Error separating overlapping objects: {e}")
            raise


    def filter_contours(self, contours, min_area=50, max_area=2000):
        """
        Filter contours based on area to exclude noise and invalid shapes.

        Args:
            contours: List of contours to filter.
            min_area: Minimum area for valid contours.
            max_area: Maximum area for valid contours.

        Returns:
            Filtered list of contours.
        """
        try:
            return [contour for contour in contours if min_area <= cv2.contourArea(contour) <= max_area]
        except Exception as e:
            self.logger.error(f"Error filtering contours: {e}")
            raise

    def filter_nested_ellipses(self, ellipses: np.ndarray) -> np.ndarray:
        """
        Remove nested or redundant ellipses based on center distance and size.

        Args:
            ellipses: Array of detected ellipses (x, y, major_axis, minor_axis, angle).

        Returns:
            Filtered array of ellipses.
        """
        try:
            filtered = []
            for i, (x1, y1, major1, minor1, angle1) in enumerate(ellipses):
                nested = False
                for j, (x2, y2, major2, minor2, angle2) in enumerate(ellipses):
                    if i != j:
                        # Compute distance between centers
                        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        
                        # Check if one ellipse is nested within another
                        if distance < (major1 / 2) and (major2 / 2) > (major1 / 2):
                            nested = True
                            break
                if not nested:
                    filtered.append((x1, y1, major1, minor1, angle1))
            return np.array(filtered)
        except Exception as e:
            self.logger.error(f"Error filtering nested ellipses: {e}")
            raise

    def non_max_suppression(self, ellipses: np.ndarray, overlap_threshold: float = 0.5) -> np.ndarray:
        """
        Apply non-maximum suppression to remove overlapping ellipses.

        Args:
            ellipses: Array of detected ellipses (x, y, major_axis, minor_axis, angle).
            overlap_threshold: Overlap threshold for suppression (0.0 to 1.0).

        Returns:
            Array of ellipses after suppression.
        """
        try:
            if len(ellipses) == 0:
                return ellipses

            # Convert ellipses to rectangles for IoU calculation
            rects = [
                (x - major / 2, y - minor / 2, x + major / 2, y + minor / 2)
                for x, y, major, minor, _ in ellipses
            ]
            indices = cv2.dnn.NMSBoxes(rects, [1] * len(rects), 0.0, overlap_threshold)
            return ellipses[indices.flatten()] if len(indices) > 0 else np.array([])
        except Exception as e:
            self.logger.error(f"Error in non-maximum suppression: {e}")
            raise




class ResultAnalyzer: # Post processing stats and probabilities to be used for better detection in the future (ML)
    """
    Analyzes and validates M&M detection results, generating statistics and reports.
    
    Attributes:
        logger: Logging instance for error tracking
        history: List of previous results for trend analysis
        validation_thresholds: Dictionary of validation thresholds
    """
    
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize result history
        self.history: List[Dict] = []
        
        # Define validation thresholds
        self.validation_thresholds = {
            'min_count': 1,
            'max_count': 50,
            'min_radius': 8,
            'max_radius': 35,
            'min_distance': 15,
            'max_overlap': 0.3
        }
        
    def analyze_results(self, circles: np.ndarray, 
                       colors: List[str],
                       timestamp: Optional[datetime.datetime] = None) -> Dict:
        """
        Analyze detection results and generate statistics.
        
        Args:
            circles: Detected circles
            colors: Identified colors
            timestamp: Time of detection
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Validate inputs
            if len(circles) != len(colors):
                raise ValueError("Mismatch between circles and colors count")
                
            # Create result dictionary
            results = {
                'timestamp': timestamp or datetime.datetime.now().isoformat(),
                'total_count': len(circles),
                'color_distribution': dict(Counter(colors)),
                'spatial_statistics': self._calculate_spatial_statistics(circles),
                'radius_statistics': self._calculate_radius_statistics(circles),
                'validation_results': self._validate_results(circles, colors),
                'confidence_metrics': self._calculate_confidence_metrics(circles, colors)
            }
            
            # Add to history
            self.history.append(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in result analysis: {str(e)}")
            raise
            
    def _calculate_spatial_statistics(self, circles: np.ndarray) -> Dict:
        """
        Calculate spatial statistics for detected M&Ms.
        
        Args:
            circles: Detected circles
            
        Returns:
            Dictionary containing spatial statistics
        """
        try:
            if len(circles) == 0:
                return {'density': 0, 'average_spacing': 0}
                
            # Extract center points
            centers = circles[:, :2]
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)
                    
            # Calculate area
            area = (np.max(centers[:, 0]) - np.min(centers[:, 0])) * \
                   (np.max(centers[:, 1]) - np.min(centers[:, 1]))
                   
            stats = {
                'density': len(circles) / area if area > 0 else 0,
                'average_spacing': np.mean(distances) if distances else 0,
                'min_spacing': np.min(distances) if distances else 0,
                'max_spacing': np.max(distances) if distances else 0,
                'spacing_std': np.std(distances) if distances else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial statistics: {str(e)}")
            raise
            
    def _calculate_radius_statistics(self, circles: np.ndarray) -> Dict:
        """
        Calculate statistics for M&M radii.
        
        Args:
            circles: Detected circles
            
        Returns:
            Dictionary containing radius statistics
        """
        try:
            if len(circles) == 0:
                return {
                    'average_radius': 0,
                    'radius_std': 0,
                    'min_radius': 0,
                    'max_radius': 0
                }
                
            radii = circles[:, 2]
            
            stats = {
                'average_radius': float(np.mean(radii)),
                'radius_std': float(np.std(radii)),
                'min_radius': float(np.min(radii)),
                'max_radius': float(np.max(radii)),
                'radius_variation': float(np.std(radii) / np.mean(radii))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating radius statistics: {str(e)}")
            raise
            
    def _calculate_confidence_metrics(self, circles: np.ndarray, 
                                   colors: List[str]) -> Dict:
        """
        Calculate confidence metrics for detection results.
        
        Args:
            circles: Detected circles
            colors: Identified colors
            
        Returns:
            Dictionary containing confidence metrics
        """
        try:
            metrics = {
                'detection_confidence': 0.0,
                'color_confidence': 0.0,
                'overall_confidence': 0.0
            }
            
            if len(circles) == 0:
                return metrics
                
            # Detection confidence based on radius consistency
            radii = circles[:, 2]
            radius_variation = np.std(radii) / np.mean(radii)
            detection_confidence = max(0, 1 - radius_variation)
            
            # Color confidence based on known colors
            known_colors = len([c for c in colors if c != 'unknown'])
            color_confidence = known_colors / len(colors) if colors else 0
            
            # Overall confidence
            metrics.update({
                'detection_confidence': float(detection_confidence),
                'color_confidence': float(color_confidence),
                'overall_confidence': float((detection_confidence + color_confidence) / 2)
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence metrics: {str(e)}")
            raise
            
    def _validate_results(self, circles: np.ndarray, colors: List[str]) -> Dict:
        """
        Validate detection results against thresholds.
        
        Args:
            circles: Detected circles
            colors: Identified colors
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation = {
                'count_valid': True,
                'radius_valid': True,
                'spacing_valid': True,
                'overlap_valid': True,
                'warnings': []
            }
            
            # Validate count
            if len(circles) < self.validation_thresholds['min_count']:
                validation['count_valid'] = False
                validation['warnings'].append("Too few M&Ms detected")
            elif len(circles) > self.validation_thresholds['max_count']:
                validation['count_valid'] = False
                validation['warnings'].append("Too many M&Ms detected")
                
            # Validate radii
            radii = circles[:, 2]
            if np.any(radii < self.validation_thresholds['min_radius']) or \
               np.any(radii > self.validation_thresholds['max_radius']):
                validation['radius_valid'] = False
                validation['warnings'].append("Invalid M&M sizes detected")
                
            # Validate spacing
            centers = circles[:, :2]
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < self.validation_thresholds['min_distance']:
                        validation['spacing_valid'] = False
                        validation['warnings'].append("M&Ms too close together")
                        break
                        
            return validation
            
        except Exception as e:
            self.logger.error(f"Error in result validation: {str(e)}")
            raise
            
    def generate_report(self, results: Dict) -> str:
        """
        Generate a formatted report of analysis results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted report string
        """
        try:
            report = [
                "M&M Detection Analysis Report",
                f"Timestamp: {results['timestamp']}",
                f"\nTotal M&Ms detected: {results['total_count']}",
                "\nColor Distribution:"
            ]
            
            for color, count in results['color_distribution'].items():
                report.append(f"  {color}: {count}")
                
            report.extend([
                "\nSpatial Statistics:",
                f"  Average spacing: {results['spatial_statistics']['average_spacing']:.2f} pixels",
                f"  Density: {results['spatial_statistics']['density']:.4f} M&Ms/pixel²",
                "\nSize Statistics:",
                f"  Average radius: {results['radius_statistics']['average_radius']:.2f} pixels",
                f"  Radius std dev: {results['radius_statistics']['radius_std']:.2f} pixels",
                "\nConfidence Metrics:",
                f"  Detection confidence: {results['confidence_metrics']['detection_confidence']:.2%}",
                f"  Color confidence: {results['confidence_metrics']['color_confidence']:.2%}",
                f"  Overall confidence: {results['confidence_metrics']['overall_confidence']:.2%}"
            ])
            
            if results['validation_results']['warnings']:
                report.extend([
                    "\nWarnings:",
                    *[f"  - {warning}" for warning in results['validation_results']['warnings']]
                ])
                
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
            
    def plot_results(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Generate visualization plots for results.
        
        Args:
            results: Analysis results dictionary
            save_path: Optional path to save plots
        """
        try:
            plt.style.use('seaborn')
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Color distribution
            ax1 = plt.subplot(2, 2, 1)
            colors = list(results['color_distribution'].keys())
            counts = list(results['color_distribution'].values())
            ax1.bar(colors, counts)
            ax1.set_title('Color Distribution')
            ax1.set_xlabel('Color')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Radius distribution
            ax2 = plt.subplot(2, 2, 2)
            ax2.hist(results['radius_statistics']['average_radius'], bins=10)
            ax2.set_title('Radius Distribution')
            ax2.set_xlabel('Radius (pixels)')
            ax2.set_ylabel('Frequency')
            
            # Confidence metrics
            ax3 = plt.subplot(2, 2, 3)
            confidence_metrics = results['confidence_metrics']
            ax3.bar(['Detection', 'Color', 'Overall'], 
                   [confidence_metrics['detection_confidence'],
                    confidence_metrics['color_confidence'],
                    confidence_metrics['overall_confidence']])
            ax3.set_title('Confidence Metrics')
            ax3.set_ylim(0, 1)
            
            # Spatial distribution
            ax4 = plt.subplot(2, 2, 4)
            spatial_stats = results['spatial_statistics']
            ax4.bar(['Min', 'Avg', 'Max'],
                   [spatial_stats['min_spacing'],
                    spatial_stats['average_spacing'],
                    spatial_stats['max_spacing']])
            ax4.set_title('Spacing Distribution')
            ax4.set_ylabel('Pixels')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
            raise
            
    # Saving the results allows the model to pull on the saved data to further improve the model
    def save_results(self, results: Dict, filepath: str) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Analysis results dictionary
            filepath: Path to save file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
                
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
            
    def get_history_statistics(self) -> Dict:
        """
        Calculate statistics across historical results.
        
        Returns:
            Dictionary containing historical statistics
        """
        try:
            if not self.history:
                return {}
                
            total_counts = [result['total_count'] for result in self.history]
            
            stats = {
                'average_count': np.mean(total_counts),
                'count_std': np.std(total_counts),
                'min_count': min(total_counts),
                'max_count': max(total_counts),
                'total_analyses': len(self.history),
                'color_frequencies': self._calculate_color_frequencies()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating history statistics: {str(e)}")
            raise
            
    def _calculate_color_frequencies(self) -> Dict:
        """
        Calculate color frequencies across all historical results.
        
        Returns:
            Dictionary containing color frequencies
        """
        try:
            if not self.history:
                return {}
                
            all_colors = []
            for result in self.history:
                all_colors.extend(result['color_distribution'].keys())
                
            return dict(Counter(all_colors))
            
        except Exception as e:
            self.logger.error(f"Error calculating color frequencies: {str(e)}")
            raise

    def evaluate_detections(self, detected: List[str], ground_truth: List[str]) -> Dict:
        """
        Evaluate detection performance against ground truth.
        Args:
            detected: List of detected object labels.
            ground_truth: List of actual object labels.
        Returns:
            Performance metrics: Precision, Recall, F1-Score.
        """
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score

            detected_labels = [1 if d != 'unknown' else 0 for d in detected]
            ground_truth_labels = [1 for _ in ground_truth]  # Assume all ground truth are valid M&Ms.

            precision = precision_score(ground_truth_labels, detected_labels)
            recall = recall_score(ground_truth_labels, detected_labels)
            f1 = f1_score(ground_truth_labels, detected_labels)

            self.logger.info(f"Evaluation metrics - Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
            return {'precision': precision, 'recall': recall, 'f1_score': f1}
        except Exception as e:
            self.logger.error(f"Error evaluating detections: {e}")
            raise



class MMCounterGUI: # All this is the GUI that allows effective interaction with the model and images with ease
    """
    Main GUI application for M&M Counter System.
    
    Integrates all components:
    - WebcamCapture for image acquisition
    - ImageProcessor for image processing
    - MMDetector for M&M detection
    - ResultAnalyzer for result analysis
    """
    
    def __init__(self):
        """Initialize the GUI application and all components."""
        self.setup_logging()
        self.setup_components()
        self.setup_gui()
        
    def setup_logging(self):
        """Configure logging system."""
        self.logger = logging.getLogger(__name__)
        
    def setup_components(self):
        """Initialize all system components."""
        try:
            self.webcam = WebcamCapture()
            self.processor = ImageProcessor()
            self.detector = MMDetector()
            self.analyzer = ResultAnalyzer()
            
            # Initialize state variables
            self.current_image = None
            self.current_results = None
            self.is_processing = False
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
        
    def setup_gui(self):
        """Set up the graphical user interface."""
        try:
            self.root = tk.Tk()
            self.root.title("M&M Counter System")
            self.root.geometry("1200x800")
            
            # Create main container with padding
            self.main_container = ttk.Frame(self.root, padding="10")
            self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            self.main_container.columnconfigure(1, weight=1)
            self.main_container.rowconfigure(1, weight=1)
            
            self.setup_control_panel()
            self.setup_image_display()
            self.setup_results_panel()
            self.setup_status_bar()
            
        except Exception as e:
            self.logger.error(f"Error setting up GUI: {str(e)}")
            raise

    def setup_control_panel(self):
        """Set up the control panel with buttons and options."""
        try:
            # Create main control frame
            control_frame = ttk.LabelFrame(self.main_container, text="Controls", padding="5")
            control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
            
            # File operations
            file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding="5")
            file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
            
            ttk.Button(file_frame, text="Load Image", 
                      command=self.load_image).grid(row=0, column=0, padx=2)
            ttk.Button(file_frame, text="Save Results", 
                      command=self.save_results).grid(row=0, column=1, padx=2)
            
            # Camera operations
            camera_frame = ttk.LabelFrame(control_frame, text="Camera Control", padding="5")
            camera_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
            
            ttk.Button(camera_frame, text="Start Camera", 
                      command=self.start_camera).grid(row=0, column=0, padx=2)
            ttk.Button(camera_frame, text="Capture", 
                      command=self.capture_from_webcam).grid(row=0, column=1, padx=2)
            ttk.Button(camera_frame, text="Stop Camera", 
                      command=self.stop_camera).grid(row=0, column=2, padx=2)
            
            # Processing controls
            process_frame = ttk.LabelFrame(control_frame, text="Processing", padding="5")
            process_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
            
            ttk.Button(process_frame, text="Process Image", 
                      command=self.process_image).grid(row=0, column=0, padx=2)
            ttk.Button(process_frame, text="Clear", 
                      command=self.clear_display).grid(row=0, column=1, padx=2)
            
            # Settings
            settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding="5")
            settings_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

            # Add fields for ellipse detection parameters
            settings_frame = ttk.LabelFrame(self.main_container, text="Ellipse Detection Settings", padding="5")
            settings_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

            ttk.Label(settings_frame, text="Min Radius:").grid(row=0, column=0, padx=2)
            self.min_radius_var = tk.StringVar(value=str(self.detector.min_radius))
            ttk.Entry(settings_frame, textvariable=self.min_radius_var, width=5).grid(row=0, column=1, padx=2)

            ttk.Label(settings_frame, text="Max Radius:").grid(row=0, column=2, padx=2)
            self.max_radius_var = tk.StringVar(value=str(self.detector.max_radius))
            ttk.Entry(settings_frame, textvariable=self.max_radius_var, width=5).grid(row=0, column=3, padx=2)

            ttk.Label(settings_frame, text="Min Ratio:").grid(row=1, column=0, padx=2)
            self.min_ratio_var = tk.StringVar(value=str(self.detector.min_ratio))
            ttk.Entry(settings_frame, textvariable=self.min_ratio_var, width=5).grid(row=1, column=1, padx=2)

            ttk.Label(settings_frame, text="Max Ratio:").grid(row=1, column=2, padx=2)
            self.max_ratio_var = tk.StringVar(value=str(self.detector.max_ratio))
            ttk.Entry(settings_frame, textvariable=self.max_ratio_var, width=5).grid(row=1, column=3, padx=2)

            ttk.Button(settings_frame, text="Apply", command=self.apply_ellipse_settings).grid(row=2, column=0, columnspan=4, pady=5)

            # Analysis options
            analysis_frame = ttk.LabelFrame(control_frame, text="Analysis Options", padding="5")
            analysis_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
            
            self.show_colors_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(analysis_frame, text="Show Colors", 
                           variable=self.show_colors_var).grid(row=0, column=0)
            
            self.show_centers_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(analysis_frame, text="Show Centers", 
                           variable=self.show_centers_var).grid(row=0, column=1)
            
        except Exception as e:
            self.logger.error(f"Error setting up control panel: {str(e)}")
            raise

    def setup_image_display(self):
        """Set up the image display area."""
        try:
            display_frame = ttk.LabelFrame(self.main_container, text="Image Display", padding="5")
            display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
            
            self.image_label = ttk.Label(display_frame)
            self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Configure grid weights
            display_frame.columnconfigure(0, weight=1)
            display_frame.rowconfigure(0, weight=1)
        except Exception as e:
            self.logger.error(f"Error setting up image display: {str(e)}")
            raise

    def setup_results_panel(self):
        """Set up the results display panel."""
        try:
            results_frame = ttk.LabelFrame(self.main_container, text="Results", padding="5")
            results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
            
            # Results text display with scrollbar
            self.results_text = tk.Text(results_frame, height=10, width=40)
            scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                    command=self.results_text.yview)
            
            self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            self.results_text['yscrollcommand'] = scrollbar.set
            
            # Configure grid weights
            results_frame.columnconfigure(0, weight=1)
            results_frame.rowconfigure(0, weight=1)
        except Exception as e:
            self.logger.error(f"Error setting up results panel: {str(e)}")
            raise

    def setup_status_bar(self):
        """Set up the status bar."""
        try:
            self.status_var = tk.StringVar()
            self.status_var.set("Ready")
            
            status_bar = ttk.Label(self.main_container, textvariable=self.status_var, 
                                 relief=tk.SUNKEN)
            status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        except Exception as e:
            self.logger.error(f"Error setting up status bar: {str(e)}")
            raise
        
    def apply_ellipse_settings(self):
        """Apply detection parameter settings."""
        try:
            '''
            # Update detector parameters
            min_radius = int(self.min_radius_var.get())
            max_radius = int(self.max_radius_var.get())
            
            self.detector.adjust_detection_params(
                min_radius=min_radius,
                max_radius=max_radius
            )
            
            self.status_var.set("Detection parameters updated")
            '''
            # Fetch parameters from GUI
            min_radius = float(self.min_radius_var.get())
            max_radius = float(self.max_radius_var.get())
            min_ratio = float(self.min_ratio_var.get())
            max_ratio = float(self.max_ratio_var.get())

            # Update detector parameters
            self.detector.set_detection_params(min_radius=min_radius, max_radius=max_radius,
                                            min_ratio=min_ratio, max_ratio=max_ratio)

            self.status_var.set("Ellipse detection parameters updated.")
            self.logger.info("Ellipse detection parameters applied successfully.")
            
            # Reprocess current image if available
            if self.current_image is not None:
                self.process_image()
                
        except ValueError as e:
            messagebox.showerror("Error", "Invalid parameter values")
            self.logger.error(f"Invalid ellipse settings: {e}")
            
        except Exception as e:
            self.logger.error(f"Error applying settings: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")

    def load_image(self):
        """Load image from file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("Failed to load image")
                    
                self.display_image(self.current_image)
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
                self.logger.info(f"Loaded image from {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def save_results(self):
        """Save current results to file."""
        try:
            if self.current_results is None:
                messagebox.showwarning("Warning", "No results to save")
                return
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            
            if file_path:
                self.analyzer.save_results(self.current_results, file_path)
                self.status_var.set(f"Results saved to {os.path.basename(file_path)}")
                
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def start_camera(self):
        """Initialize and start the webcam preview."""
        try:
            if self.webcam.initialize_camera():
                self.preview_window = tk.Toplevel(self.root)
                self.preview_window.title("Camera Preview")
                self.preview_label = ttk.Label(self.preview_window)
                self.preview_label.pack()
                
                self.update_preview()
                self.status_var.set("Camera preview active")
            else:
                messagebox.showerror("Error", "Failed to initialize camera")
        except Exception as e:
            self.logger.error(f"Error starting camera: {str(e)}")
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")

    def update_preview(self):
        """Update the camera preview."""
        if hasattr(self, 'preview_window'):
            frame = self.webcam.capture_frame()
            if frame is not None:
                self.display_image(frame, preview=True)
                self.root.after(10, self.update_preview)

    def capture_from_webcam(self):
        """Capture image from webcam."""
        try:
            frame = self.webcam.capture_frame()
            if frame is not None:
                self.current_image = frame
                self.display_image(frame)
                self.status_var.set("Image captured from webcam")
            else:
                messagebox.showerror("Error", "Failed to capture image")
        except Exception as e:
            self.logger.error(f"Error capturing from webcam: {str(e)}")
            messagebox.showerror("Error", f"Failed to capture image: {str(e)}")

    def stop_camera(self):
        """Stop the webcam preview and release resources."""
        try:
            if hasattr(self, 'preview_window'):
                self.preview_window.destroy()
                delattr(self, 'preview_window')
                
            self.webcam.release()
            self.status_var.set("Camera stopped")
        except Exception as e:
            self.logger.error(f"Error stopping camera: {str(e)}")
            messagebox.showerror("Error", f"Failed to stop camera: {str(e)}")

    def process_image(self):
        """Process the current image and detect M&Ms."""
        try:
            if self.current_image is None:
                messagebox.showwarning("Warning", "No image loaded")
                return
                
            if self.is_processing:
                return
                
            self.is_processing = True
            self.status_var.set("Processing image...")
            
            # Process image
            gray, hsv, blurred = self.processor.process_for_detection(self.current_image)
            
            # Detect M&Ms
            circles, colors = self.detector.process_image(gray, hsv)
            
            if len(circles) == 0:
                self.status_var.set("No M&Ms detected")
                return
                
            # Analyze results
            self.current_results = self.analyzer.analyze_results(circles, colors)
            
            # Display results
            self.display_results(self.current_results)
            
            # Display processed image
            result_image = self.detector.draw_results(self.current_image, circles, colors)
            self.display_image(result_image)
            
            self.status_var.set(f"Detected {len(circles)} M&Ms")
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            
        finally:
            self.is_processing = False
    
    def display_image(self, image: np.ndarray, preview: bool = False):
        """Display image in GUI."""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Resize if needed
            display_size = (800, 600)
            pil_image.thumbnail(display_size, Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display
            if preview:
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo
            else:
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                
        except Exception as e:
            self.logger.error(f"Error displaying image: {str(e)}")
            raise

    def display_results(self, results: Dict):
        """Display analysis results in GUI."""
        try:
            report = self.analyzer.generate_report(results)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, report)
        except Exception as e:
            self.logger.error(f"Error displaying results: {str(e)}")
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")

    def clear_display(self):
        """Clear the current image and results."""
        self.current_image = None
        self.current_results = None
        self.image_label.configure(image='')
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Display cleared")

    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, 'webcam'):
                self.webcam.release()
        except:
            pass

# Create and run the application
if __name__ == "__main__":
    try:
        app = MMCounterGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {str(e)}")