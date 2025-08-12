import io
from flask import Flask, request, jsonify
import pdfplumber
import fitz 
import os
import tempfile
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import pytesseract
import re
import numpy as np
import cv2
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from functools import lru_cache
from invoice2data import extract_data

# Initialize Flask app once
app = Flask(__name__)
# Set tesseract path explicitly if needed
tesseract_cmd = os.environ.get('TESSERACT_PATH', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
# Load models lazily using lru_cache to avoid reloading
@lru_cache(maxsize=1)
def get_donut_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model.eval()
    return processor, model

# Utility function for file handling
def save_uploaded_file(file):
    """Save uploaded file to temp location and return filepath"""
    if not file or not file.filename:
        return None
    
    filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        file.save(tmp_file.name)
        return tmp_file.name
    return None

# OCR functions
def run_ocr_on_pdf(filepath, first_page_only=False):
    """Extract text from PDF using OCR"""
    try:
        if first_page_only:
            images = convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        else:
            images = convert_from_path(filepath, dpi=300)
            
        if not images:
            return "", []

        full_text = ''
        lines = []
        
        for page in images:
            text = pytesseract.image_to_string(page)
            full_text += text + '\n'
            lines += [line.strip() for line in text.split('\n') if line.strip()]
            
        return full_text, lines
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return "", []

# Table extraction functions
def extract_table_from_pdf(filepath, column_format=False):
    """Extract tables from PDF using pdfplumber"""
    table_data = []
    column_data = {}
    
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                        
                    headers = [h.strip() if h else f"Column{i}" for i, h in enumerate(table[0])]
                    
                    # Initialize column lists if using column format
                    if column_format and not column_data:
                        for header in headers:
                            column_data[header] = []
                    
                    for row in table[1:]:
                        if len(row) < len(headers):
                            continue
                            
                        # Row format (list of dicts)
                        row_dict = {}
                        for i, cell in enumerate(row):
                            if i < len(headers):
                                col_name = headers[i]
                                cell_value = cell.strip() if cell else None
                                row_dict[col_name] = cell_value
                                
                                # Also add to column data if using that format
                                if column_format:
                                    column_data[col_name].append(cell_value)
                                    
                        if row_dict:
                            table_data.append(row_dict)
        
        return column_data if column_format else table_data
    except Exception as e:
        print(f"Table extraction error: {str(e)}")
        return {} if column_format else []

def extract_table_with_ocr_grid(filepath, column_format=False):
    """Extract tables using OCR and CV2 grid detection"""
    try:
        images = convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        if not images:
            return {} if column_format else []

        img = np.array(images[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

        # Detect lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        table_mask = cv2.add(detect_horizontal, detect_vertical)

        # Find cells
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 30 or h < 20:
                continue
            cell_img = img[y:y+h, x:x+w]
            text = pytesseract.image_to_string(cell_img, config='--psm 6').strip()
            if text:
                cells.append((x, y, text))

        # Group by rows
        rows = {}
        for x, y, text in cells:
            row_key = y // 20
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append((x, text))

        # Sort rows
        table = []
        for key in sorted(rows):
            row = sorted(rows[key], key=lambda x: x[0])
            table.append([col[1] for col in row])

        if not table or len(table) < 2:
            return {} if column_format else []

        # Process table data
        headers = table[0]
        
        if column_format:
            # Column format (dict of lists)
            column_data = {header: [] for header in headers}
            for row in table[1:]:
                if len(row) < len(headers):
                    continue
                for i, cell_value in enumerate(row):
                    if i < len(headers):
                        column_data[headers[i]].append(cell_value)
            return column_data
        else:
            # Row format (list of dicts)
            structured = []
            for row in table[1:]:
                if len(row) < len(headers):
                    continue
                structured.append(dict(zip(headers, row)))
            return structured
    
    except Exception as e:
        print(f"OCR table extraction error: {str(e)}")
        return {} if column_format else []

# Text extraction
def extract_text_blocks(filepath):
    """Extract text blocks using PyMuPDF"""
    try:
        doc = fitz.open(filepath)
        page = doc[0]
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1])  # sort top-down
        return page, blocks
    except Exception as e:
        print(f"Text block extraction error: {str(e)}")
        return None, []

# Routes

@app.route("/", methods=["GET", "HEAD"])
def home():
    return "Server is running", 200


@app.route('/extract_donut', methods=['POST'])
def extract_with_donut():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400

    filepath = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Failed to save file'}), 500

    try:
        # Convert PDF to image
        images = convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        if not images:
            return jsonify({'error': 'PDF conversion failed'}), 500

        # Get model
        processor, model = get_donut_model()
        
        # Process image
        image = images[0].convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Run inference
        task_prompt = "<s_docvqa><s_question>Extract all key information from this invoice</s_question><s_answer>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)

        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        result_json = processor.token2json(result)

        return jsonify(result_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/extract_invoicenet', methods=['POST'])
def extract_with_invoicenet():
    print("extract_invoicenet called")
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        print("File is not a PDF")
        return jsonify({'error': 'Only PDF files allowed'}), 400

    filepath = save_uploaded_file(file)
    if not filepath:
        print("Failed to save file")
        return jsonify({'error': 'Failed to save file'}), 500

    try:
        text, lines = run_ocr_on_pdf(filepath)

        # Get column format if specified in request
        column_format = request.args.get('column_format', 'false').lower() == 'true'
        table = extract_table_from_pdf(filepath, column_format=column_format)

        def extract_field(pattern, default=None):
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    return line
            return default

        data = {
            "companyName": extract_field(r"company\s*name|firm\s*name|organisation|organization|inc\.?|incorporated|"
                                            r"pvt\s*ltd|private\s*limited|limited|ltd\.?|llp|llc|co\.?|corporation|corp\.?|"
                                            r"enterprise|enterprises|industries|industry|group|solutions|technologies|"
                                            r"rich\s+products|trading\s+co|traders|associates|manufacturers|"
                                            r"services\s+pvt|global\s+ltd|international|systems|engineering|"
                                            r"consultancy|consultants|corporate\s+house"),
            "companyAddress": extract_field(
                                            r"address|addr\.?|location|premises|building|bldg\.?|floor|flr\.?|block|"
                                            r"road|rd\.?|street|st\.?|avenue|ave\.?|lane|ln\.?|sector|area|"
                                            r"colony|market|complex|circle|plaza|gali|nag(a|ar)|marg|"
                                            r"district|city|state|zip|pincode|postal\s*code|\d{5,6}"
                                        ),
            "phoneNumber": extract_field(r'(?:phone|ph|ph:|Ph: |tel|telephone|contact|mobile|cell|mob)[\s\.:\-]*(?:\+?\d|\(\d)'),
            "vendor": {
                "vendorCode": extract_field(r'vendor\s*code'),
                "nameAndAddress": extract_field(r'vendor name|consulting|corporate house')
            },
            "paymentAdvice": {
                "paymentAdviceNo": extract_field(
                    r"(?:payment\s*advice\s*(?:no|number)|advice\s*(?:no|number)|advice\s*id|"
                    r"payment\s*ref(?:erence)?|advice\s*ref(?:erence)?)\s*[:\-]?\s*([A-Za-z0-9\/\-\_]+)"
                ),
                "date": extract_field(
                    r"(?:date|invoice\s*date|payment\s*date|issue\s*date|dt\.?)\s*[:\-]?\s*"
                    r"(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}|\d{4}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{1,2})"
                ),
                "chequeNo": extract_field(
                    r"(?:cheque\s*(?:no|number)|check\s*(?:no|number)|chq\s*(?:no|number))\s*[:\-]?\s*([A-Za-z0-9\-\/]+)"
                ),
                "bank": extract_field(
                    r"(?:bank\s*(?:name)?|banking|account\s*name|beneficiary\s*bank)\s*[:\-]?\s*([A-Za-z0-9&\.,\-\s]+)"
                ),
                "accountNo": extract_field(
                    r"(?:account\s*(?:no|number)|a\/c\s*(?:no|number)|iban|acct\s*(?:no|number))\s*[:\-]?\s*(\w+)"
                ),
                "paymentAmount": extract_field(
                    r"(?:amount\s*payable|total\s*amount|amt\s*payable|payment\s*amount|amt)\s*[:\-]?\s*([\₹\$\€]?\d[\d,]*(?:\.\d{1,2})?)"
                ),
                "utrNumber": extract_field(
                    r"(?:utr\s*(?:no|number)|transaction\s*(?:id|no|number)|ref(?:erence)?\s*(?:no|number))\s*[:\-]?\s*(\w+)"
                )
            },

            "table": table
        }

       

        return jsonify(data)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/extract', methods=['POST'])
def extract_pdf_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400

    filepath = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Failed to save file'}), 500

    # Get column format if specified in request
    column_format = request.args.get('column_format', 'false').lower() == 'true'
    
    data = {
        'companyName': '',
        'address': '',
        'right_vendor': '',
        'table': {} if column_format else []
    }

    try:
        # Try PyMuPDF + pdfplumber first
        page_blocks = extract_text_blocks(filepath)
        if page_blocks and page_blocks[0]:
            page, blocks = page_blocks
            text_lines = [b[4].strip() for b in blocks if b[4].strip()]
            data['companyName'] = "\n".join(text_lines[:2])

            address, right_vendor = [], []
            for block in blocks[2:6]:
                x0, x1 = block[0], block[2]
                text = block[4].strip()
                if x1 < page.rect.width / 2:
                    address.append(text)
                else:
                    right_vendor.append(text)
            data['address'] = "\n".join(address)
            data['right_vendor'] = "\n".join(right_vendor)

            table_data = extract_table_from_pdf(filepath, column_format=column_format)
            if table_data:
                data['table'] = table_data
                return jsonify(data)

        # OCR Fallback
        ocr_text, lines = run_ocr_on_pdf(filepath, first_page_only=True)
        if not lines:
            return jsonify({'error': 'OCR failed to extract text'}), 500

        data['companyName'] = "\n".join(lines[:2])
        data['address'] = "\n".join(lines[2:4])
        data['right_vendor'] = "\n".join(lines[4:6])
        data['table'] = extract_table_with_ocr_grid(filepath, column_format=column_format)

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            
@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/invoice2data',methods=['POST'])
def invoice2data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name
        result = extract_data(tmp_path)
        print(result)
        return jsonify({'text': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400

    filepath = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Failed to save file'}), 500

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        all_text = ""
        for page in doc:
            page_text = page.get_text()
            all_text += page_text + "\n"
        doc.close()
        # Replace actual line breaks with literal "\n" for JSON
        all_text = all_text.replace('\n', '\\n')
        return jsonify({"text": all_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
