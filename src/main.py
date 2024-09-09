import argparse
from utils import main
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess piece detector")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing image files")
    args = parser.parse_args()

    # Ensure the directory exists
    if not os.path.isdir(args.directory_path):
        print(f"Error: The directory {args.directory_path} does not exist.")
        exit(1)

    # List all files in the directory
    all_files = os.listdir(args.directory_path)

    # Filter and process each image file
    for filename in all_files:
            image_path = os.path.join(args.directory_path, filename)
            print(f"Processing {image_path}")
            main(image_path, filename)
