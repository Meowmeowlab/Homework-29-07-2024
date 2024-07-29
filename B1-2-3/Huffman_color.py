from PIL import Image
import numpy as np
import pickle
import heapq
from collections import defaultdict, Counter


class HuffmanNode:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


def build_huffman_tree(frequency):
    heap = [HuffmanNode(value, freq) for value, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.frequency + node2.frequency)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]


def build_codes(node, prefix="", codebook={}):
    if node is not None:
        if node.value is not None:
            codebook[node.value] = prefix
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook


def huffman_compress(data):
    frequency = Counter(data)
    huffman_tree = build_huffman_tree(frequency)
    codebook = build_codes(huffman_tree)

    encoded_data = ''.join(codebook[char] for char in data)
    return encoded_data, huffman_tree


def huffman_decompress(encoded_data, huffman_tree):
    decoded_data = []
    node = huffman_tree
    for bit in encoded_data:
        node = node.left if bit == '0' else node.right
        if node.value is not None:
            decoded_data.append(node.value)
            node = huffman_tree
    return ''.join(decoded_data)

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
    encoded_data, huffman_tree = huffman_compress(image_string)

    with open(bin_path, 'wb') as bin_file:
        pickle.dump((encoded_data, huffman_tree,
                    image.size, image.mode), bin_file)

# Decode .bin file to image


def decode_bin_to_image(bin_path, output_image_path):
    with open(bin_path, 'rb') as bin_file:
        encoded_data, huffman_tree, size, mode = pickle.load(bin_file)

    decoded_data = huffman_decompress(encoded_data, huffman_tree)
    decoded_image = string_to_image(decoded_data, size, mode)
    decoded_image.save(output_image_path)


# Example usage
if __name__ == "__main__":
    input_image_path = './testimg/meow_900p.bmp'
    compressed_bin_path = './output/compressed_huffman_image.bin'
    output_image_path = './output/output_huffman_image.bmp'

    encode_image_to_bin(input_image_path, compressed_bin_path)
    decode_bin_to_image(compressed_bin_path, output_image_path)
