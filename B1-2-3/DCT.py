import numpy as np
from PIL import Image
import cv2
import pickle

# DCT and IDCT


def dct_2d(matrix):
    return cv2.dct(np.float32(matrix))


def idct_2d(matrix):
    return cv2.idct(np.float32(matrix))

# Quantization and Dequantization


def quantize(block, q_matrix):
    return np.round(block / q_matrix)


def dequantize(block, q_matrix):
    return block * q_matrix

# Zigzag Scan


def zigzag(input_matrix):
    h, w = input_matrix.shape
    result = np.empty(h * w, dtype=input_matrix.dtype)
    index = -1
    bound = h + w - 1

    for k in range(bound):
        if k % 2 == 0:
            i = min(k, h - 1)
            while i >= 0 and k - i < w:
                index += 1
                result[index] = input_matrix[i, k - i]
                i -= 1
        else:
            j = min(k, w - 1)
            while j >= 0 and k - j < h:
                index += 1
                result[index] = input_matrix[k - j, j]
                j -= 1

    return result


def inverse_zigzag(input_vector, h, w):
    result = np.empty((h, w), dtype=input_vector.dtype)
    index = -1
    bound = h + w - 1

    for k in range(bound):
        if k % 2 == 0:
            i = min(k, h - 1)
            while i >= 0 and k - i < w:
                index += 1
                result[i, k - i] = input_vector[index]
                i -= 1
        else:
            j = min(k, w - 1)
            while j >= 0 and k - j < h:
                index += 1
                result[k - j, j] = input_vector[index]
                j -= 1

    return result

# Image to Blocks and Blocks to Image


def image_to_blocks(image):
    blocks = []
    width, height = image.size
    for i in range(0, width, 8):
        for j in range(0, height, 8):
            block = image.crop((i, j, i + 8, j + 8))
            blocks.append(np.array(block))
    return blocks


def blocks_to_image(blocks, width, height):
    image = Image.new('L', (width, height))
    k = 0
    for i in range(0, width, 8):
        for j in range(0, height, 8):
            block = Image.fromarray(np.uint8(blocks[k]))
            image.paste(block, (i, j))
            k += 1
    return image


# Quantization Matrix
Q_base = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def scale_quant_matrix(quality):
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    q_matrix = np.floor((Q_base * scale + 50) / 100)
    q_matrix[q_matrix == 0] = 1  # prevent division by zero
    return q_matrix

# Encode image to .bin file


def encode_image_to_bin(image_path, bin_path, quality=50):
    image = Image.open(image_path).convert('L')
    width, height = image.size
    blocks = image_to_blocks(image)
    Q = scale_quant_matrix(quality)

    dct_blocks = []
    for block in blocks:
        dct_block = dct_2d(block - 128)
        q_block = quantize(dct_block, Q)
        zigzag_block = zigzag(q_block)
        dct_blocks.append(zigzag_block)

    with open(bin_path, 'wb') as bin_file:
        pickle.dump((dct_blocks, width, height), bin_file)

# Decode .bin file to image


def decode_bin_to_image(bin_path, output_image_path, quality=50):
    with open(bin_path, 'rb') as bin_file:
        dct_blocks, width, height = pickle.load(bin_file)

    Q = scale_quant_matrix(quality)
    blocks = []
    for dct_block in dct_blocks:
        q_block = inverse_zigzag(dct_block, 8, 8)
        dequant_block = dequantize(q_block, Q)
        idct_block = idct_2d(dequant_block) + 128
        blocks.append(np.clip(idct_block, 0, 255).astype(np.uint8))

    image = blocks_to_image(blocks, width, height)
    image.save(output_image_path)


# Example usage
if __name__ == "__main__":
    input_image_path = './testimg/meow_900p.bmp'
    compressed_bin_path = './output/compressed_dct_image.bin'
    output_image_path = './output/output_dct_image.bmp'
    # Adjust the quality between 1 (worst) to 95 (best)
    compression_quality = 95

    encode_image_to_bin(
        input_image_path, compressed_bin_path, compression_quality)
    decode_bin_to_image(compressed_bin_path, output_image_path)
