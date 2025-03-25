import cv2
import pytesseract
import pandas as pd
import re
import os

# Set Tesseract OCR Path (Windows users must update this path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change if needed

# File to store business details
EXCEL_FILE = "business_cards.xlsx"

def extract_text_from_image(image):
    """Extract text from the visiting card image using OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    text = pytesseract.image_to_string(gray)
    return text

def get_business_name_and_phone(extracted_text):
    """Extract business name (first line) and valid phone number from text"""
    lines = extracted_text.split("\n")  # Split text into lines
    business_name = lines[0].strip() if lines else "Unknown"  # First line as Business Name

    # Regex to find valid 10-digit phone numbers starting with 6, 7, 8, or 9
    phone_pattern = r'\b[6789]\d{9}\b'
    phone_numbers = re.findall(phone_pattern, extracted_text)

    phone_number = phone_numbers[0] if phone_numbers else "Not Found"  # First valid number
    return business_name, phone_number

def save_to_excel(data):
    """Save extracted business details to an Excel file"""
    if not data:
        print("⚠️ No business details found. Excel file NOT created.")
        return

    try:
        df = pd.DataFrame(data, columns=["Business Name", "Phone Number"])  # Create DataFrame
        df.to_excel(EXCEL_FILE, index=False)  # Save as Excel file
        print(f"✅ Business details saved to '{EXCEL_FILE}' successfully!")
    except Exception as e:
        print("❌ Error saving Excel file:", e)

def capture_and_process():
    """Capture an image from the webcam and extract business details"""
    cap = cv2.VideoCapture(0)  # Open webcam
    business_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture image")
            break

        # Show instructions on screen
        cv2.putText(frame, "Press 'C' to Capture | 'Q' to Quit", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Business Card Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture image
            print("📸 Capturing image...")
            text = extract_text_from_image(frame)
            business_name, phone_number = get_business_name_and_phone(text)
            business_data.append([business_name, phone_number])
            print(f"✅ Extracted Business Name: {business_name}")
            print(f"📞 Extracted Phone Number: {phone_number}")

        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save extracted details
    if business_data:
        save_to_excel(business_data)

# Run the real-time scanner
capture_and_process()

# Open the saved Excel file automatically (Windows)
if os.path.exists(EXCEL_FILE):
    os.system(f"start {EXCEL_FILE}")  # Opens the file
