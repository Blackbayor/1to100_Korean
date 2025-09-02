import shutil
import sys

dir_path = sys.argv[1]

try:
    shutil.rmtree(dir_path)
    print(f"Successfully deleted directory: {dir_path}")
except FileNotFoundError:
    print(f"Directory not found: {dir_path}", file=sys.stderr)
except Exception as e:
    print(f"Error deleting directory {dir_path}: {e}", file=sys.stderr)