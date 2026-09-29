FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# gcc: build tools for some Python wheels
# tesseract-ocr: OCR for scanned/image-only PDFs and images (pytesseract)
# tesseract-ocr-eng: English language data (default, avoids extra langs)
# poppler-utils: provides pdftoppm used by pdf2image for OCR fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids multi-GB CUDA wheels that OOM 512MB instances)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p /app/chromadb /app/uploads

EXPOSE 8000

# Render assigns a dynamic $PORT; bind to it or fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
