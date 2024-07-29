from PIL import Image
import numpy as np
import pickle

# LZW Compression


def lzw_compress(uncompressed):
    """Compress a string to a list of output symbols."""
    # Build the dictionary.
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    p = ""
    result = []
    for c in uncompressed:
        pc = p + c
        if pc in dictionary:
            p = pc
        else:
            result.append(dictionary[p])
            dictionary[pc] = dict_size
            dict_size += 1
            p = c
    if p:
        result.append(dictionary[p])
    return result

# LZW Decompression


def lzw_decompress(compressed):
    """Decompress a list of output symbols to a string."""
    # Build the dictionary.
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}

    w = result = chr(compressed.pop(0))
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result += entry

        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1

        w = entry
    return result

# Convert image to string


def image_to_string(image):
    return image.tobytes().decode('latin1')

# Convert string to image


def string_to_image(s, size, mode):
    return Image.frombytes(mode, size, s.encode('latin1'))

# Encode image to .bin file


def encode_image_to_bin(image_path, bin_path):
    image = Image.open(image_path)
    image_string = image_to_string(image)
    compressed_data = lzw_compress(image_string)
    with open(bin_path, 'wb') as bin_file:
        pickle.dump((compressed_data, image.size, image.mode), bin_file)

# Decode .bin file to image


def decode_bin_to_image(bin_path, output_image_path):
    with open(bin_path, 'rb') as bin_file:
        compressed_data, size, mode = pickle.load(bin_file)
    decompressed_data = lzw_decompress(compressed_data)
    decompressed_image = string_to_image(decompressed_data, size, mode)
    decompressed_image.save(output_image_path)


# Example usage
if __name__ == "__main__":
    input_image_path = './testimg/meow_900p.bmp'
    compressed_bin_path = './output/compressed_lzw_image.bin'
    output_image_path = './output/output_lzw_image.bmp'

    encode_image_to_bin(input_image_path, compressed_bin_path)
    decode_bin_to_image(compressed_bin_path, output_image_path)
