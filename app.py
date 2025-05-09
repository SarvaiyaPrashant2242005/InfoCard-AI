from flask import Flask, render_template, request, send_file, redirect, url_for, flash, make_response
import easyocr
import re
import os
import pandas as pd
import spacy
import numpy as np
from werkzeug.utils import secure_filename
import logging
import uuid
from datetime import datetime
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = "uploads"
EXCEL_FILE = "business_cards.xlsx"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'bmp'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize EasyOCR and spaCy
try:
    logger.info("Initializing EasyOCR reader")
    reader = easyocr.Reader(['en'])
    logger.info("Loading spaCy model")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_font_size(box):
    """Calculate approximate font size based on bounding box dimensions"""
    # Extract the coordinates of the bounding box
    # The format is typically [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # We calculate the height of the bounding box
    height = max(
        abs(box[0][1] - box[2][1]),
        abs(box[1][1] - box[3][1])
    )
    return height

def extract_info(image_path):
    """Extract information from business card image using font size detection"""
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Perform OCR with detail=1 to get bounding boxes
        results = reader.readtext(image_path, detail=1)
        
        if not results:
            logger.warning("No text detected in the image")
            return {"error": "No text detected in the image"}
        
        # Sort results by vertical position (top to bottom)
        results.sort(key=lambda x: x[0][0][1])  # Sort by y-coordinate of first point
        
        # Extract information with font size analysis
        font_sizes = []
        lines = []
        line_positions = []
        line_boxes = []
        
        for detection in results:
            box = detection[0]  # bounding box coordinates
            text = detection[1]  # the text string
            
            # Skip very short texts (likely noise or punctuation)
            if len(text.strip()) <= 1:
                continue
                
            # Calculate font size based on bounding box height
            font_size = calculate_font_size(box)
            font_sizes.append(font_size)
            lines.append(text.strip())
            
            # Store the vertical position (y-coordinate)
            line_positions.append(box[0][1])
            line_boxes.append(box)
        
        # Normalize font sizes to a scale of 1-10
        if font_sizes:
            font_size_array = np.array(font_sizes)
            min_size = np.min(font_size_array)
            max_size = np.max(font_size_array)
            
            # Prevent division by zero
            if max_size > min_size:
                normalized_sizes = 1 + 9 * (font_size_array - min_size) / (max_size - min_size)
            else:
                normalized_sizes = np.ones_like(font_size_array) * 5
                
            # Create a list of tuples (text, font_size, position)
            text_with_sizes = list(zip(lines, normalized_sizes, line_positions, line_boxes))
            
            # Log detected text with sizes for debugging
            for text, size, pos, _ in text_with_sizes:
                logger.debug(f"Text: '{text}', Size: {size:.2f}, Position: {pos}")
            
            # Form full text for general processing
            full_text = " ".join(lines)
            
            # Initialize extracted data
            extracted_data = {}
            
            # BUSINESS NAME EXTRACTION LOGIC
            # Method 1: Use the text with largest font size
            business_name_candidates = []
            
            # Find lines with largest font sizes (above threshold)
            size_threshold = 7.5  # Adjust this threshold as needed
            large_font_texts = [(text, size) for text, size, _, _ in text_with_sizes if size > size_threshold]
            
            if large_font_texts:
                # Sort by font size in descending order
                large_font_texts.sort(key=lambda x: x[1], reverse=True)
                business_name_candidates.append(large_font_texts[0][0])
            
            # Method 2: Look at the first few lines (traditional approach)
            top_lines = [text for text, _, _, _ in text_with_sizes[:3] if len(text) > 5]
            if top_lines:
                business_name_candidates.append(top_lines[0])
            
            # Method 3: Look for organizational entities with spaCy
            doc = nlp(full_text)
            org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if org_entities:
                business_name_candidates.append(org_entities[0])
            
            # Method 4: Look for text that might be centered on the card
            # Calculate horizontal center of the image
            img_center_x = img_width / 2
            
            # Find text that's centered on the card
            centered_texts = []
            for text, size, _, box in text_with_sizes:
                if len(text) > 5:  # Minimum text length for a business name
                    # Calculate text center x position
                    text_center_x = (box[0][0] + box[1][0]) / 2
                    # Check if text is near the center
                    if abs(text_center_x - img_center_x) < (img_width * 0.2):  # Within 20% of center
                        centered_texts.append((text, size))
            
            if centered_texts:
                # Sort by font size in descending order
                centered_texts.sort(key=lambda x: x[1], reverse=True)
                business_name_candidates.append(centered_texts[0][0])
            
            # Check for business indicators in the string
            business_indicators = ['inc', 'llc', 'ltd', 'corp', 'gmbh', 'co', 'company', 'enterprises', 
                                  'group', 'services', 'solutions', 'technologies', 'associates']
            
            for text, size, _, _ in text_with_sizes:
                if any(indicator in text.lower() for indicator in business_indicators):
                    business_name_candidates.append(text)
                    break
            
            # Remove duplicates and sort by priority
            business_name_candidates = list(dict.fromkeys(business_name_candidates))
            
            # Select the best business name from candidates
            if business_name_candidates:
                extracted_data["Business Name"] = business_name_candidates[0]
            else:
                extracted_data["Business Name"] = "Not Found"
            
            # EXTRACT PERSON NAME
            # Use spaCy Named Entity Recognition
            person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            
            # Look for text that might include common name patterns
            name_patterns = [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b'  # First M. Last
            ]
            
            name_matches = []
            for pattern in name_patterns:
                matches = re.findall(pattern, full_text)
                name_matches.extend(matches)
            
            # Combine with spaCy results and remove duplicates
            person_candidates = list(dict.fromkeys(person_entities + name_matches))
            
            # Filter out business name from person names
            if extracted_data["Business Name"] != "Not Found":
                person_candidates = [name for name in person_candidates 
                                    if name != extracted_data["Business Name"]]
            
            extracted_data["Person Name"] = ", ".join(person_candidates) if person_candidates else "Not Found"
            
            # If no person name found, try second line as fallback
            if extracted_data["Person Name"] == "Not Found" and len(lines) > 1:
                # Check if second line is not already used as business name
                if lines[1] != extracted_data["Business Name"]:
                    extracted_data["Person Name"] = lines[1]
            
            # Extract phone numbers
            phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
            phone_numbers = re.findall(phone_pattern, full_text)
            extracted_data["Phone Number"] = ", ".join(phone_numbers) if phone_numbers else "Not Found"
            
            # Extract emails
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            emails = re.findall(email_pattern, full_text)
            extracted_data["Email"] = ", ".join(emails) if emails else "Not Found"
            
            # Extract website
            website_pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?'
            websites = re.findall(website_pattern, full_text)
            extracted_data["Website"] = ", ".join(websites) if websites else "Not Found"
            
            # Extract address components
            address_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
            
            # Look for postal codes
            postal_code_pattern = r'\b\d{5}(?:-\d{4})?\b'  # US postal code pattern
            postal_codes = re.findall(postal_code_pattern, full_text)
            
            # Look for address patterns
            address_patterns = [
                r'\d+\s+[A-Za-z0-9\s,]+(?:Avenue|Ave|Boulevard|Blvd|Street|St|Road|Rd|Lane|Ln|Drive|Dr|Way|Court|Ct|Plaza|Plz|Place|Pl)\b',
                r'\d+\s+[A-Za-z0-9\s,]+(?:Suite|Ste|Floor|Fl|Unit|Apt|Apartment)\s+\d+\b'
            ]
            
            address_matches = []
            for pattern in address_patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                address_matches.extend(matches)
            
            # Combine all address components
            address_components = address_entities + postal_codes + address_matches
            extracted_data["Address"] = ", ".join(address_components) if address_components else "Not Found"
            
            # Store raw text for verification
            extracted_data["Raw Text"] = full_text
            
            # Store font size information for verification
            extracted_data["Font Analysis"] = ", ".join([f"{line} (size: {size:.2f})" for line, size, _, _ in text_with_sizes[:5]])
            
            # Add timestamp
            extracted_data["Processed Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info("Successfully extracted information")
            return extracted_data
        else:
            logger.warning("No valid text lines detected in the image")
            return {"error": "No valid text detected in the image"}
        
    except Exception as e:
        logger.error(f"Error extracting information: {str(e)}")
        return {"error": f"Error processing image: {str(e)}"}

def save_to_excel(new_data):
    """Save extracted data to Excel file"""
    try:
        df_new = pd.DataFrame([new_data])
        
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined.to_excel(EXCEL_FILE, index=False)
        logger.info(f"Data saved to {EXCEL_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving to Excel: {str(e)}")
        return False

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for handling file uploads and displaying results"""
    if request.method == "POST":
        # Check if the post request has the file part
        if 'card' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['card']
        
        # If user does not select file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Generate unique filename to prevent overwriting
            original_filename = secure_filename(file.filename)
            filename = f"{uuid.uuid4()}_{original_filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the file
            file.save(filepath)
            logger.info(f"File saved: {filepath}")
            
            # Process the image
            result = extract_info(filepath)
            
            # Check for errors
            if "error" in result:
                flash(result["error"])
                return render_template("index.html")
                
            # Save to Excel
            if save_to_excel(result):
                return render_template("index.html", result=result, filename=original_filename)
            else:
                flash("Error saving data to Excel file")
                return render_template("index.html")
        else:
            flash(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(request.url)
            
    return render_template("index.html")

@app.route("/download")
def download_excel():
    """Route for downloading the Excel file"""
    if os.path.exists(EXCEL_FILE):
        return send_file(EXCEL_FILE, as_attachment=True)
    else:
        flash("No data available for download")
        return redirect(url_for('index'))

@app.route("/view")
def view_data():
    """Route for viewing all extracted data"""
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        records = df.to_dict('records')
        return render_template("view.html", records=records)
    else:
        flash("No data available to view")
        return redirect(url_for('index'))

@app.route("/search", methods=["POST"])
def search_data():
    """Route for searching business card data"""
    if request.method == "POST":
        search_term = request.form.get("search_term", "").lower()
        search_fields = request.form.getlist("search_fields")
        
        if not os.path.exists(EXCEL_FILE):
            flash("No data available to search")
            return redirect(url_for('view_data'))
        
        df = pd.read_excel(EXCEL_FILE)
        
        # Filter based on search term and fields
        if search_term:
            mask = pd.Series(False, index=df.index)  # Initialize with all False
            
            for field in search_fields:
                if field in df.columns:
                    # Apply filter to this column
                    field_mask = df[field].astype(str).str.lower().str.contains(search_term, na=False)
                    mask = mask | field_mask
                
            df_filtered = df[mask]
        else:
            df_filtered = df
        
        # Convert filtered dataframe to records
        records = df_filtered.to_dict('records')
        
        return render_template("view.html", records=records, search_term=search_term)
    
    return redirect(url_for('view_data'))

@app.route("/delete/<int:row_id>", methods=["POST"])
def delete_record(row_id):
    """Route for deleting a specific record"""
    try:
        if not os.path.exists(EXCEL_FILE):
            flash("No data available")
            return redirect(url_for('view_data'))
        
        df = pd.read_excel(EXCEL_FILE)
        
        if 0 <= row_id < len(df):
            # Delete the row
            df = df.drop(index=row_id).reset_index(drop=True)
            df.to_excel(EXCEL_FILE, index=False)
            flash("Record successfully deleted")
        else:
            flash("Invalid record ID")
            
    except Exception as e:
        logger.error(f"Error deleting record: {str(e)}")
        flash(f"Error deleting record: {str(e)}")
        
    return redirect(url_for('view_data'))

@app.route("/export_vcard/<int:row_id>")
def export_vcard(row_id):
    """Route for exporting a record as vCard"""
    try:
        if not os.path.exists(EXCEL_FILE):
            flash("No data available")
            return redirect(url_for('view_data'))
        
        df = pd.read_excel(EXCEL_FILE)
        
        if 0 <= row_id < len(df):
            # Get the record
            record = df.iloc[row_id]
            
            # Create vCard content
            vcard = [
                "BEGIN:VCARD",
                "VERSION:3.0",
                f"FN:{record['Person Name']}",
                f"ORG:{record['Business Name']}",
                f"TEL;TYPE=WORK:{record['Phone Number']}",
                f"EMAIL:{record['Email']}",
                f"URL:{record['Website']}" if record['Website'] != "Not Found" else "",
                f"ADR;TYPE=WORK:;;{record['Address']}" if record['Address'] != "Not Found" else "",
                "END:VCARD"
            ]
            
            # Filter out empty lines
            vcard = [line for line in vcard if line]
            
            # Create response
            response = make_response("\n".join(vcard))
            response.headers["Content-Disposition"] = f"attachment; filename={record['Person Name'].replace(' ', '_')}.vcf"
            response.headers["Content-Type"] = "text/vcard"
            return response
        else:
            flash("Invalid record ID")
            
    except Exception as e:
        logger.error(f"Error exporting vCard: {str(e)}")
        flash(f"Error exporting vCard: {str(e)}")
        
    return redirect(url_for('view_data'))

if __name__ == "__main__":
    logger.info("Starting Business Card Scanner application")
    app.run(debug=True)