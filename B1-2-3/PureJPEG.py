from PIL import Image
import pickle

# Encode image to .bin file with adjustable compression quality


def encode_image_to_bin(image_path, bin_path, quality):
    # Open the image
    image = Image.open(image_path)

    # Save the image as a JPEG with the specified quality
    temp_jpeg_path = 'temp_image.jpg'
    image.save(temp_jpeg_path, format='JPEG', quality=quality)

    # Read the JPEG image
    with open(temp_jpeg_path, 'rb') as jpeg_file:
        jpeg_data = jpeg_file.read()

    # Save the JPEG data to a binary file
    with open(bin_path, 'wb') as bin_file:
        pickle.dump(jpeg_data, bin_file)

# Decode .bin file to image


def decode_bin_to_image(bin_path, output_image_path):
    # Read the JPEG data from the binary file
    with open(bin_path, 'rb') as bin_file:
        jpeg_data = pickle.load(bin_file)

    # Save the JPEG data as an image file
    temp_jpeg_path = 'temp_image.jpg'
    with open(temp_jpeg_path, 'wb') as jpeg_file:
        jpeg_file.write(jpeg_data)

    # Open the saved JPEG file and save it in the desired output format
    image = Image.open(temp_jpeg_path)
    image.save(output_image_path)


# Example usage
if __name__ == "__main__":
    input_image_path = './testimg/meow_900p.bmp'
    compressed_bin_path = './output/compressed_jpeg_image.bin'
    output_image_path = './output/output_jpeg_image.jpeg'

    # Adjust the quality between 1 (worst) to 95 (best)
    compression_quality = 50

    encode_image_to_bin(
        input_image_path, compressed_bin_path, compression_quality)
    decode_bin_to_image(compressed_bin_path, output_image_path)
