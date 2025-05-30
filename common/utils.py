"""
Common utilities for Orbbec camera operations.
This module provides functions for converting different color formats to BGR for OpenCV compatibility.
"""

from typing import Union, Any, Optional
import cv2
import numpy as np
from pyorbbecsdk import FormatConvertFilter, VideoFrame, OBFormat, OBConvertFormat

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    """
    Convert a VideoFrame to a BGR image format compatible with OpenCV.
    
    Args:
        frame: VideoFrame from the Orbbec camera
        
    Returns:
        np.array: BGR formatted image or None if conversion fails
    """
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print(f"Unsupported color format: {color_format}")
        return None
    return image

def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert I420 format to BGR"""
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)

def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert NV12 format to BGR"""
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert NV21 format to BGR"""
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21) 