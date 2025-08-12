# Pdf Invoice To json


## TO start this project locally
```
git clone https://github.com/dhwanijain-dev/v2InvoiceApi.git
```
 
## API Endpoints

### `/`  
**Method:** `GET`  
Health check route. Returns `"Server is running"` if the API is up.

---

### `/extract_donut`  
**Method:** `POST`  
**Request:** PDF file (as form-data, key: `file`)

Extracts key information from the first page of the uploaded PDF using the Donut deep learning model (DocVQA).  
**Returns:** JSON object with extracted fields as predicted by the Donut model.

---

### `/extract_invoicenet`  
**Method:** `POST`  
**Request:** PDF file (as form-data, key: `file`), optional query param: `column_format=true` for columnar table output

Performs OCR on the PDF and extracts invoice-related fields (company name, address, phone, vendor, payment advice, etc.) using regex patterns. Also extracts tables using pdfplumber.  
**Returns:** JSON object with structured invoice data and table(s).

---

### `/extract`  
**Method:** `POST`  
**Request:** PDF file (as form-data, key: `file`), optional query param: `column_format=true` for columnar table output

Extracts company name, address, vendor, and tables from the PDF using PyMuPDF and pdfplumber. Falls back to OCR if text extraction fails.  
**Returns:** JSON object with company info and table(s).

---

### `/ocr`  
**Method:** `POST`  
**Request:** JSON body: `{ "image": "<base64-encoded-image>" }`

Performs OCR on a base64-encoded image and returns the extracted text.  
**Returns:** JSON object: `{ "text": "..." }`

---

### `/invoice2data`  
**Method:** `POST`  
**Request:** PDF file (as form-data, key: `file`)

Uses the invoice2data library to extract invoice fields from the uploaded PDF.  
**Returns:** JSON object: `{ "text": ... }` with extracted invoice data.

---

### `/extract_text`  
**Method:** `POST`  
**Request:** PDF file (as form-data, key: `file`)

Extracts all text from the PDF (like Ctrl+A), preserving line breaks as `\n`.  
**Returns:** JSON object: `{ "text": "..." }` with all PDF text, line breaks as `\n`.

