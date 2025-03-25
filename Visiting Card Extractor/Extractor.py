import cv2
import pytesseract
import pandas as pd
import re

# Configure Tesseract path (Make sure this path is correct)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def extract_info(text):
    """Extracts name, email, phone, and address from OCR text."""
    name = text.split('\n')[0]  # Assuming first line is name
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.findall(r'\+?\d{10,13}', text)
    address = '\n'.join(text.split('\n')[1:])  # Rest of text as address
   
    return {
        'Name': name.strip(),
        'Email': email[0] if email else '',
        'Phone': phone[0] if phone else '',
        'Address': address.strip()
    }

def capture_card():
    """Captures an image from the webcam and saves it as 'visiting_card.jpg'."""
    print("Press 's' to capture the visiting card.")
    
    cap = cv2.VideoCapture(0)  # Initialize webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Check your webcam.")
            break
        
        cv2.imshow("Visiting Card Scanner", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("visiting_card.jpg", frame)
            print("Image captured successfully!")
            break

    cap.release()
    cv2.destroyAllWindows()
    return "visiting_card.jpg"

def process_card(image_path):
    """Processes the captured image using OCR and extracts information."""
    if not image_path:
        print("No image captured.")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to load image.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    
    return extract_info(text)

def save_to_excel(data, filename="E:/Projects/visiting_cards.xlsx"):
    """Saves extracted information to an Excel file."""
    if not data:
        print("No data to save.")
        return

    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Email', 'Phone', 'Address'])
   
    df = df.append(data, ignore_index=True)
    df.to_excel(filename, index=False)
    print("Data saved to", filename)

if __name__ == "__main__":
    image_path = capture_card()
    extracted_data = process_card(image_path)
    
    if extracted_data:
        print("Extracted Data:", extracted_data)
        save_to_excel(extracted_data)
    else:
        print("No data extracted.")
