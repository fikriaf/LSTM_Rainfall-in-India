import PyPDF2


def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text


if __name__ == "__main__":
    pdf_path = "Proposal Project Kelompok_Prediksi Curah Hujan Harian di India Menggunakan Metode Long Short-Term Memory (LSTM).pdf"
    text = extract_pdf_text(pdf_path)
    print(text)
