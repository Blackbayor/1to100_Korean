import fitz  # PyMuPDF
import os
import shutil

def convert_pdfs_to_pngs_skip_first_page():
    """
    Converts all PDF files in the specified input directory to PNG images,
    skipping the first page of each PDF.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_dir = os.path.join(project_root, 'data', 'raw_0805')
    output_dir = os.path.join(project_root, 'data', 'processed', 'images_0805')
    
    # Create or clear the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared existing directory: {output_dir}")
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

    try:
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_filename)
        try:
            pdf_document = fitz.open(pdf_path)
        except fitz.fitz.FitzError as e:
            print(f"Could not open {pdf_filename}: {e}")
            continue

        base_filename = os.path.splitext(pdf_filename)[0]
        print(f"Processing {pdf_filename}...")

        # Iterate through each page of the PDF, skipping the first page
        for page_num in range(1, len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Render page to an image (pixmap) with 72 DPI
            zoom = 72 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            output_image_path = os.path.join(output_dir, f"{base_filename}_page_{page_num + 1}.png")
            
            pix.save(output_image_path)
            
        print(f"  > Finished converting {len(pdf_document) - 1} pages (skipped first page).")
        pdf_document.close()

    print("\nConversion complete.")

if __name__ == "__main__":
    convert_pdfs_to_pngs_skip_first_page()
