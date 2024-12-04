# Required imports for the M&M Counter System
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
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

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WebcamCapture:
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
        self.camera_settings = {
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

class ImageProcessor:
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
    """
    Handles M&M detection and color classification.
    
    Attributes:
        logger: Logging instance for error tracking
        min_radius: Minimum radius for M&M detection
        max_radius: Maximum radius for M&M detection
        min_dist: Minimum distance between detected circles
        param1: First parameter for Hough Circle detection
        param2: Second parameter for Hough Circle detection
        color_ranges: HSV color ranges for M&M classification
    """
    
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        







        # Circle detection parameters
        self.min_radius = 10
        self.max_radius = 30
        self.min_dist = 20
        self.param1 = 50  # Edge detection parameter
        self.param2 = 30  # Circle detection parameter
        
        # Define color ranges in HSV format
        self.color_ranges = {
            'red': {
                'lower1': np.array([0, 100, 100]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]),  # Red wraps around HSV
                'upper2': np.array([180, 255, 255])
            },
            'blue': {
                'lower': np.array([100, 100, 100]),
                'upper': np.array([130, 255, 255])
            },
            'green': {
                'lower': np.array([40, 100, 100]),
                'upper': np.array([80, 255, 255])
            },
            'yellow': {
                'lower': np.array([20, 100, 100]),
                'upper': np.array([35, 255, 255])
            },
            'orange': {
                'lower': np.array([10, 100, 100]),
                'upper': np.array([20, 255, 255])
            },
            'brown': {
                'lower': np.array([0, 100, 20]),
                'upper': np.array([20, 255, 100])
            }
        }
        # Detection parameters history for analysis
        self.detection_history = []
        
    def detect_circles(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect circles in the grayscale image using Hough Circle Transform.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Array of detected circles (x, y, radius)
        """
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
            
    def identify_colors(self, hsv: np.ndarray, circles: np.ndarray) -> List[str]:
        """
        Identify colors of detected M&Ms.
        
        Args:
            hsv: HSV format image
            circles: Detected circles
            
        Returns:
            List of color names for each detected M&M
        """
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
            
    def _classify_color(self, hsv_color: Tuple[float, float, float]) -> str:
        """
        Classify color based on HSV values.
        
        Args:
            hsv_color: HSV color values
            
        Returns:
            Identified color name
        """
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
                              param2: Optional[int] = None) -> None:
        """
        Adjust circle detection parameters.
        
        Args:
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            min_dist: Minimum distance between circles
            param1: Edge detection parameter
            param2: Circle detection parameter
        """
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
            
    def get_parameters(self) -> Dict:
        """
        Get current detection parameters.
        
        Returns:
            Current parameter values
        """
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
            gray: Grayscale image
            hsv: HSV format image
            
        Returns:
            Tuple containing detected circles and their colors
        """
        try:
            # Detect circles
            circles = self.detect_circles(gray)
            
            # Identify colors if circles were found
            colors = self.identify_colors(hsv, circles) if len(circles) > 0 else []
            
            return circles, colors
            
        except Exception as e:
            self.logger.error(f"Error in image processing: {str(e)}")
            raise
            
    def draw_results(self, image: np.ndarray, circles: np.ndarray, 
                    colors: List[str]) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Original image
            circles: Detected circles
            colors: Identified colors
            
        Returns:
            Image with detection results drawn
        """
        try:
            result_image = image.copy()
            
            for (x, y, r), color in zip(circles, colors):
                # Draw circle outline
                cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)
                
                # Add color label
                cv2.putText(result_image, color, (x-20, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            return result_image
            
        except Exception as e:
            self.logger.error(f"Error drawing results: {str(e)}")
            raise

class ResultAnalyzer:
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

class MMCounterGUI:
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
            
            # Detection parameters
            ttk.Label(settings_frame, text="Min Radius:").grid(row=0, column=0, padx=2)
            self.min_radius_var = tk.StringVar(value=str(self.detector.min_radius))
            ttk.Entry(settings_frame, textvariable=self.min_radius_var, 
                     width=5).grid(row=0, column=1, padx=2)
            
            ttk.Label(settings_frame, text="Max Radius:").grid(row=0, column=2, padx=2)
            self.max_radius_var = tk.StringVar(value=str(self.detector.max_radius))
            ttk.Entry(settings_frame, textvariable=self.max_radius_var, 
                     width=5).grid(row=0, column=3, padx=2)
            
            ttk.Button(settings_frame, text="Apply", 
                      command=self.apply_settings).grid(row=0, column=4, padx=5)
            
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
        
    def apply_settings(self):
        """Apply detection parameter settings."""
        try:
            # Update detector parameters
            min_radius = int(self.min_radius_var.get())
            max_radius = int(self.max_radius_var.get())
            
            self.detector.adjust_detection_params(
                min_radius=min_radius,
                max_radius=max_radius
            )
            
            self.status_var.set("Detection parameters updated")
            
            # Reprocess current image if available
            if self.current_image is not None:
                self.process_image()
                
        except ValueError as e:
            messagebox.showerror("Error", "Invalid parameter values")
            
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