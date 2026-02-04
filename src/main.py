import os
import glob
import requests
from logger import setup_logging
from importance import calculate_prioritized_runout_dates
from datetime import datetime
import pandas as pd

def clear_temp():
    """
    Clear temp files to prevent running the program with old data.
    """
    for file in glob.glob("data/temp/*"):
        try:
            os.remove(file)
        except IsADirectoryError:
            pass  # ignore folders if any

def get_formatted_date():
    """
    Gets user input for what date they want to predict for. Defaults to the present date if empty.
    """
    while True:
        date_str = input("Enter a date (MM-DD-YYYY format) or leave blank for today: ").strip()

        # If input is empty, return today's date
        if not date_str:
            return datetime.now().strftime("%m-%d-%Y")

        try:
            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
            return date_obj.strftime("%m-%d-%Y")
        except ValueError:
            print("Invalid format. Please try again (Example: 05-27-2025)")

def download_sheets(sheet_url, output_folder='data/temp'):
    """
    Downloads the inventory, production, and flavor_weights google sheet for modeling
    
    Args:
        sheet_url (str): The URL of the Google Sheet
        output_folder (str): Folder to save the CSVs (default: 'data/temp')
    """
    # Clean the URL and extract document ID
    sheet_url = sheet_url.split('/edit')[0]
    doc_id = sheet_url.split('/d/')[1]

    sheets = {
        'inventory': '1801631785',
        'production': '961276732',
        'flavor_weights' : '1153477294'
    }
    
    # Download each sheet
    for sheet_name, gid in sheets.items():
        export_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"
        
        response = requests.get(export_url)
        response.raise_for_status()
        
        output_path = os.path.join(output_folder, f"{sheet_name}.csv")
        with open(output_path, 'wb') as f:
            f.write(response.content)

# Usage
def main():
    user_date = get_formatted_date()
    setup_logging(user_date)
    print(f"Selected date: {user_date}")

    os.makedirs(f'logs/{user_date}', exist_ok=True)

    print("[1/8] Clearing temp files...")
    clear_temp()
    print("[2/8] Downloading sheets...")
    download_sheets("https://docs.google.com/spreadsheets/d/19Mq2bg3RGr2HlVtVdgtjdFLciJSMF6eCWxcnwjN63Fc/edit?usp=sharing")

    # Clean asset files
    df = pd.read_csv('data/temp/inventory.csv')
    df = df[df['Date'].notna()]
    df = df[df['Date'] != 'FALSE']
    df.to_csv('data/temp/inventory.csv', index=False)

    df = pd.read_csv('data/temp/production.csv')
    df = df[df['Date'].notna()]
    df = df[df['Date'] != 'FALSE']
    df.to_csv('data/temp/production.csv', index=False)

    prioritized_runout = calculate_prioritized_runout_dates(user_date)
    prioritized_runout.to_csv(f'logs/{user_date}/production_priorities.csv', index=False)

main()