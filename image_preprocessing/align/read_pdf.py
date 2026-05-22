import pypdf

def extract_pdf_text(pdf_path, txt_path):
    print(f"Reading PDF from: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- PAGE {idx + 1} ---\n")
            f.write(text)
            f.write("\n\n")
    print(f"Text written to: {txt_path}")

if __name__ == "__main__":
    pdf_path = "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/Preprocessing Pipeline-May 2023 1.pdf"
    txt_path = "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_main/image_preprocessing/align/pdf_text.txt"
    extract_pdf_text(pdf_path, txt_path)
