import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from collections import Counter
import heapq
from math import log2
import math


def flip_horizontal(image):
    width, height = image.size
    # Create a new image to hold the flipped result
    flipped_image = Image.new("RGB", (width, height))
    pixels = image.load()
    flipped_pixels = flipped_image.load()
    
    for x in range(width):
        for y in range(height):
            flipped_pixels[width - 1 - x, y] = pixels[x, y]
    
    return flipped_image


def flip_vertical(image):
    width, height = image.size
    # Create a new image for the flipped result
    flipped_image = Image.new("RGB", (width, height))
    pixels = image.load()
    flipped_pixels = flipped_image.load()
    
    for x in range(width):
        for y in range(height):
            flipped_pixels[x, height - 1 - y] = pixels[x, y]
    
    return flipped_image


def noise_reduction(image):
    """Apply a median filter for noise reduction."""
    img_np = np.array(image)
    filtered_img = np.zeros_like(img_np)
    
    # Assuming a 3x3 kernel
    for y in range(1, img_np.shape[0] - 1):
        for x in range(1, img_np.shape[1] - 1):
            for c in range(img_np.shape[2]):  # Assuming RGB
                # Extract the neighborhood
                neighborhood = img_np[y-1:y+2, x-1:x+2, c]
                # Replace with the median value
                filtered_img[y, x, c] = np.median(neighborhood)
    
    return Image.fromarray(filtered_img)


def color_balance(image, red_factor, green_factor, blue_factor):
    """Adjust the color balance of an image."""
    img_np = np.array(image, dtype=np.float32)
    
    # Adjust each channel
    img_np[:,:,0] *= red_factor   # Red
    img_np[:,:,1] *= green_factor # Green
    img_np[:,:,2] *= blue_factor  # Blue
    
    # Clip the values to be in the valid range [0, 255] and convert back to uint8
    np.clip(img_np, 0, 255, out=img_np)
    return Image.fromarray(img_np.astype(np.uint8))


def color_balance_warm(image):
    """Apply a warm color balance to an image."""
    return color_balance(image, 1.2, 1.1, 1.0)


def color_balance_cool(image):
    """Apply a cool color balance to an image."""
    return color_balance(image, 1.0, 1.1, 1.2)


def solarize(image, threshold=128):
    width, height = image.size
    solarized_image = Image.new("RGB", (width, height))
    pixels = image.load()
    solarized_pixels = solarized_image.load()

    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            solarized_pixels[x, y] = tuple(
                255 - channel if channel > threshold else channel
                for channel in (r, g, b)
            )

    return solarized_image


def edge_detection(image):
    width, height = image.size
    edge_image = Image.new("RGB", (width, height), "white")
    pixels = image.load()
    edge_pixels = edge_image.load()

    for x in range(width - 1):
        for y in range(height - 1):
            r, g, b = pixels[x, y]
            r_next, g_next, b_next = pixels[x + 1, y + 1]
            diff = abs(r - r_next) + abs(g - g_next) + abs(b - b_next)

            if diff > 50:  # Threshold for edge detection
                edge_pixels[x, y] = (0, 0, 0)
            else:
                edge_pixels[x, y] = (255, 255, 255)

    return edge_image


def vignette(image, strength=1.5):
    from math import sqrt
    width, height = image.size
    vignette_image = image.copy()
    pixels = vignette_image.load()
    
    max_distance = sqrt((width / 2)**2 + (height / 2)**2)

    for x in range(width):
        for y in range(height):
            # Calculate the distance to the center of the image
            dx = x - width / 2
            dy = y - height / 2
            distance = sqrt(dx**2 + dy**2)
            # Ratio of the distance to the maximum distance
            ratio = distance / max_distance
            # The darkening factor increases with the ratio
            darkening_factor = int(255 * (ratio ** strength))
            # Retrieve the current pixel value
            r, g, b = pixels[x, y]
            # Apply the darkening effect
            pixels[x, y] = max(0, r - darkening_factor), max(0, g - darkening_factor), max(0, b - darkening_factor)

    return vignette_image



def thresholding(image, threshold=128):
    gray_image = image.convert("L")  # Convert to grayscale
    binary_image = gray_image.point(lambda x: 255 if x > threshold else 0, '1')
    return binary_image


def create_gaussian_kernel(size=3, sigma=1.0):
    """Create a square Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2
    
    if sigma == 0:
        sigma = ((size-1)*0.5 - 1)*0.3 + 0.8

    s = 2.0 * sigma * sigma

    # Calculate Gaussian function for each kernel element
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (math.exp(-(x**2 + y**2) / s)) / (math.pi * s)
    
    return kernel / np.sum(kernel)  # Normalize the kernel

def gaussian_blur_manual(image, kernel_size=3, sigma=1.0):
    """Apply Gaussian blur to an image."""
    kernel = create_gaussian_kernel(kernel_size, sigma)
    width, height = image.size
    img_np = np.array(image)
    pad_size = kernel_size // 2
    img_padded = np.pad(img_np, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    blurred_img = np.zeros_like(img_np)
    
    # Convolve the kernel with the image
    for y in range(height):
        for x in range(width):
            for c in range(3):  # Assuming RGB
                blurred_img[y, x, c] = np.sum(
                    kernel * img_padded[y:y+kernel_size, x:x+kernel_size, c])
    
    return Image.fromarray(np.uint8(blurred_img))