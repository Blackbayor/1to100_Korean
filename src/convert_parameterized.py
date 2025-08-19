import fitz  # PyMuPDF
import os
import sys
import argparse

def convert(input_dir, output_dir, dpi):
    """
    Converts all PDF files in the input directory to PNG images with a specified DPI.
    """
    # Ensure absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.join(project_root, input_dir)
    output_path = os.path.join(project_root, output_dir)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # Check if input directory exists
    if not os.path.isdir(input_path):
        print(f"Error: Input directory not found at {input_path}")
        return

    pdf_files = [f for f in os.listdir(input_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process in '{input_path}'.")

    # Process each PDF file
    for pdf_filename in pdf_files:
        pdf_filepath = os.path.join(input_path, pdf_filename)
        try:
            pdf_document = fitz.open(pdf_filepath)
        except fitz.fitz.FitzError as e:
            print(f"Could not open {pdf_filename}: {e}. Skipping.")
            continue

        base_filename = os.path.splitext(pdf_filename)[0]
        print(f"Processing {pdf_filename}...")

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            
            output_image_path = os.path.join(output_path, f"{base_filename}_page_{page_num + 1}.png")
            pix.save(output_image_path)
            
        print(f"  > Finished converting {len(pdf_document)} pages to {dpi} DPI.")
        pdf_document.close()

    print(f"\nConversion complete for '{input_dir}'. PNGs are in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF files to PNG images with specified DPI.")
    parser.add_argument("input_dir", help="Relative path to the directory containing PDF files.")
    parser.add_argument("output_dir", help="Relative path to the directory to save PNG files.")
    parser.add_argument("dpi", type=int, help="Dots Per Inch (DPI) for the output images.")
    
    args = parser.parse_args()
    
    convert(args.input_dir, args.output_dir, args.dpi)
