from PIL import Image
import pickle

# RLE Compression


def rle_compress(data):
    compressed_data = []
    count = 1
    prev_char = data[0]

    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            compressed_data.append((prev_char, count))
            prev_char = char
            count = 1
    compressed_data.append((prev_char, count))
    return compressed_data

# RLE Decompression


def rle_decompress(compressed_data):
    decompressed_data = []
    for char, count in compressed_data:
        decompressed_data.append(char * count)
    return ''.join(decompressed_data)

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
    compressed_data = rle_compress(image_string)

    with open(bin_path, 'wb') as bin_file:
        pickle.dump((compressed_data, image.size, image.mode), bin_file)

# Decode .bin file to image


def decode_bin_to_image(bin_path, output_image_path):
    with open(bin_path, 'rb') as bin_file:
        compressed_data, size, mode = pickle.load(bin_file)

    decompressed_data = rle_decompress(compressed_data)
    decompressed_image = string_to_image(decompressed_data, size, mode)
    decompressed_image.save(output_image_path)


# Example usage
if __name__ == "__main__":
    input_image_path = './testimg/meow_900p.bmp'
    compressed_bin_path = './output/compressed_RLE_image.bin'
    output_image_path = './output/output_RLE_image.bmp'

    encode_image_to_bin(input_image_path, compressed_bin_path)
    decode_bin_to_image(compressed_bin_path, output_image_path)
