
import io
import time 
import os
import concurrent.futures
from tqdm import tqdm
import re
import PyPDF2
from tabula import read_pdf

os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.awt.headless=true'

import warnings
warnings.filterwarnings("ignore")

# Path to the directory containing PDF files
pdf_root_dir = "/home/terradxllm/GAIA/TerraDX_GPT/PDF_Data"

# Output directory for text files
output_dir = "/home/terradxllm/GAIA/TerraDX_GPT/Text_Files1"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process a single PDF file
def process_pdf(pdf_path, page_number):
    text=""
    with open(pdf_path, 'rb') as file:
        reader=PyPDF2.PdfReader(file)
        if page_number<=len(reader.pages):
            page=reader.pages[page_number-1] # 0- based indexing or page numbers
            text=page.extract_text()
        else:
            print("page_number exceeded")

    tables=read_pdf(pdf_path, pages=str(page_number), multiple_tables=True)
    if tables:
        for i, table in enumerate(tables):
            text+=f"\n{table}\n"

    return text
    
def extract_text_and_tables(pdf_path):

    if not os.path.exists(pdf_path):
        print("No pdf")
    try:
        with open(pdf_path, 'rb') as file:
            reader=PyPDF2.PdfReader(file)
            pages=len(reader.pages)
    except Exception as e:
        print(f"Pdf file cannot be opened: {e}")

    extracted_text=""
    for page_num in range(1,pages+1):
            extracted_text+=f"\n{process_pdf(pdf_path, page_num)}\n"

    # Extract folder name as the file name
    folder_name = os.path.basename(os.path.dirname(pdf_path))

    # Construct output file path
    # output_file for pdf's
    folder_name_=pdf_path.split('/')[-3]
    output_file = os.path.join(output_dir, folder_name_, folder_name, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.txt")

    # Create the directory for the output file if it doesn't exist - check by exist_ok parameter
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write extracted text to the output file
    with open(output_file, "w") as f:
        f.write(extracted_text)

    print(f"PDF '{pdf_path}' converted and saved as '{output_file}'")

# Function to process PDF files recursively using multithreading
def process_pdf_files_multithreaded(root_dir):

    pdf_files = []

    # Collect all PDF files
    for root, dirs, files in os.walk(root_dir):
        # If there are any files within the directory
        if len(files)>0:
            pdf_files.extend([os.path.join(root, file) for file in files if file.endswith(".pdf")])
    
    # Process PDF files using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(extract_text_and_tables, pdf_files)

# Process PDF files in the root directory using multithreading
process_pdf_files_multithreaded(pdf_root_dir)




