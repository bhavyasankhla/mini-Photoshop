from tkinter import messagebox
from PIL import Image
import numpy as np
from collections import Counter
import heapq
from math import log2


def convert_to_grayscale_manual(image):
    width, height = image.size
    # Create a new image for the grayscale version
    grayscale_image = Image.new("L", (width, height))  # "L" mode => grayscale

    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((i, j))
            # Apply the luminosity method to calculate grayscale value
            grayscale = int(0.21 * r + 0.72 * g + 0.07 * b)
            grayscale_image.putpixel((i, j), grayscale)

    return grayscale_image


def ordered_dithering(image):    
    gray_image = convert_to_grayscale_manual(image)
    # Define a 4x4 dithering matrix
    threshold_map = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ]) * 17  # Scale the threshold map to the range of 0-255

    for y in range(0, gray_image.height):
        for x in range(0, gray_image.width):
            pixel = gray_image.getpixel((x, y))
            threshold = threshold_map[x % 4, y % 4]
            gray_image.putpixel((x, y), 255 if pixel > threshold else 0)
    return gray_image


def auto_level(image):
    img_np = np.array(image)
    if img_np.ndim == 3:  # For RGB images
        # Separate channels
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        # Apply auto-level per channel
        r_adj = np.interp(r, (r.min(), r.max()), (0, 255))
        g_adj = np.interp(g, (g.min(), g.max()), (0, 255))
        b_adj = np.interp(b, (b.min(), b.max()), (0, 255))
        # Combine channels back
        adjusted = np.stack([r_adj, g_adj, b_adj], axis=-1).astype(np.uint8)
    else:  # For grayscale images
        adjusted = np.interp(img_np, (img_np.min(), img_np.max()), (0, 255)).astype(np.uint8)
    return Image.fromarray(adjusted)


def show_huffman_info(image):    
    entropy, avg_code_length = compute_entropy_and_huffman(image)
    messagebox.showinfo("Huffman Info", f"Entropy: {entropy:.2f} bits\nAverage Huffman Code Length: {avg_code_length:.2f} bits")


def compute_entropy_and_huffman(image):
    # Convert image to grayscale manually
    grayscale_image = convert_to_grayscale_manual(image)
    
    # Flatten the image to a list of pixels
    pixels = list(grayscale_image.getdata())
    freqs = Counter(pixels)
    total_pixels = sum(freqs.values())
    
    # Calculate probabilities and entropy
    probs = {pixel: freq / total_pixels for pixel, freq in freqs.items()}
    entropy = -sum(prob * log2(prob) for prob in probs.values() if prob > 0)

    # Initial heap setup: [(probability, [(symbol, '')])]
    heap = [(prob, [(sym, '')]) for sym, prob in probs.items()]
    heapq.heapify(heap)

    # Construct Huffman Tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        new_pairs = [(sym, '0' + code) for sym, code in lo[1]] + [(sym, '1' + code) for sym, code in hi[1]]
        heapq.heappush(heap, (lo[0] + hi[0], new_pairs))

    # Calculate average Huffman code length
    if heap:
        _, symbol_code_pairs = heap[0]
        avg_code_length = sum(len(code) * probs[sym] for sym, code in symbol_code_pairs)

    return entropy, avg_code_length