import easyocr
import re
import os
import pandas as pd

reader = easyocr.Reader(['en'])

EXCEL_FILE = "output.xlsx"

def extract_info(image_path):
    results = reader.readtext(image_path, detail=0)
    lines = [line.strip() for line in results if line.strip()]

    business_name = lines[0] if lines else "Not Found"
    owner_name = lines[1] + " BHAI" if len(lines) > 1 else "Not Found"

    # ✅ Phone number regex: starts with 6/7/8/9 and 10 digits
    phone_pattern = r'(\+91[\s\-]?)?[6789]\d{9}\b'
    full_text = " ".join(lines)
    phone_match = re.search(phone_pattern, full_text)
    phone_number = phone_match.group() if phone_match else "Not Found"

    data = {
        "Business Name": business_name,
        "Owner Name": owner_name,
        "Phone Number": phone_number,
        "Raw Text": full_text
    }

    save_to_excel(data)
    return data

def save_to_excel(new_data):
    df_new = pd.DataFrame([new_data])

    if os.path.exists(EXCEL_FILE):
        df_existing = pd.read_excel(EXCEL_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_excel(EXCEL_FILE, index=False)
