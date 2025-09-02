from PIL import Image
import sys

image_path = sys.argv[1]

try:
    with Image.open(image_path) as img:
        print(img.size)
except FileNotFoundError:
    print(f"Error: File not found at {image_path}", file=sys.stderr)
except Exception as e:
    print(f"Error opening image: {e}", file=sys.stderr)