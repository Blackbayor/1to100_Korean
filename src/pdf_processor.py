import fitz  # PyMuPDF
import os
import shutil
from src.config import Config

def convert_pdfs_to_pngs(config: Config, input_dir: str):
    """
    Converts all PDF files in the input directory to PNG images, page by page.
    """
    output_dir = config.IMAGE_DIR

    # Create or clear the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared existing directory: {output_dir}")
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

    # Get a list of PDF files in the input directory
    try:
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    # Process each PDF file
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_filename)
        pdf_document = fitz.open(pdf_path)
        
        base_filename = os.path.splitext(pdf_filename)[0]
        print(f"Processing {pdf_filename}...")

        # Iterate through each page of the PDF
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Render page to an image (pixmap) with higher DPI
            zoom = config.DPI / config.PDF_STANDARD_DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Define the output image path
            output_image_path = os.path.join(output_dir, f"{base_filename}_page_{page_num + 1}.png")
            
            # Save the image
            pix.save(output_image_path)
            
        print(f"  > Finished converting {len(pdf_document)} pages.")
        pdf_document.close()

    print("\nConversion complete.")

if __name__ == "__main__":
    config = Config()
    convert_pdfs_to_pngs(config, config.RAW_DATA_DIR)
