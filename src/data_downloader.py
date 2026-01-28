import os
import requests
import pandas as pd
import numpy as np
import io
import gzip
import time
import random

class AirbnbDataDownloader:
    """
    Smart Downloader with FALLBACK Strategy.
    Priority: Dec 2025 (Q4) -> Sept 2025 (Q3) -> June 2025 (Q2) -> March 2025 (Q1).
    Attempts to get the LATEST available data for each city.
    """
    
    # Prioritized List of URLs (Newest to Oldest)
    CITY_CANDIDATES = {
        "NYC": [
            ("Q4_2025", "http://data.insideairbnb.com/united-states/ny/new-york-city/2025-12-04/data/listings.csv.gz"),
            ("Nov_2025", "http://data.insideairbnb.com/united-states/ny/new-york-city/2025-11-02/data/listings.csv.gz"),
            ("Oct_2025", "http://data.insideairbnb.com/united-states/ny/new-york-city/2025-10-06/data/listings.csv.gz"),
            ("Q3_2025", "http://data.insideairbnb.com/united-states/ny/new-york-city/2025-09-02/data/listings.csv.gz"),
            ("Q2_2025", "http://data.insideairbnb.com/united-states/ny/new-york-city/2025-06-02/data/listings.csv.gz"),
        ],
        "LA": [
            ("Q4_2025", "http://data.insideairbnb.com/united-states/ca/los-angeles/2025-12-05/data/listings.csv.gz"),
            ("Q3_2025", "http://data.insideairbnb.com/united-states/ca/los-angeles/2025-09-03/data/listings.csv.gz"),
            ("Q2_2025", "http://data.insideairbnb.com/united-states/ca/los-angeles/2025-06-17/data/listings.csv.gz"),
        ],
        "SF": [
            ("Q4_2025", "http://data.insideairbnb.com/united-states/ca/san-francisco/2025-12-30/data/listings.csv.gz"),
            ("Q3_2025", "http://data.insideairbnb.com/united-states/ca/san-francisco/2025-09-05/data/listings.csv.gz"),
            ("Q2_2025", "http://data.insideairbnb.com/united-states/ca/san-francisco/2025-06-04/data/listings.csv.gz"),
        ],
        "Chicago": [
            ("Q4_2025", "http://data.insideairbnb.com/united-states/il/chicago/2025-12-15/data/listings.csv.gz"),
            ("Q3_2025", "http://data.insideairbnb.com/united-states/il/chicago/2025-09-22/data/listings.csv.gz"),
            ("Q2_2025", "http://data.insideairbnb.com/united-states/il/chicago/2025-06-18/data/listings.csv.gz"),
        ],
        "Boston": [
            ("Q4_2025", "http://data.insideairbnb.com/united-states/ma/boston/2025-12-18/data/listings.csv.gz"),
            ("Q3_2025", "http://data.insideairbnb.com/united-states/ma/boston/2025-09-23/data/listings.csv.gz"),
            ("Q2_2025", "http://data.insideairbnb.com/united-states/ma/boston/2025-06-20/data/listings.csv.gz"),
        ]
    }

    def __init__(self, output_dir="Artifacts"):
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, "Raw_Smart_Download")
        os.makedirs(self.raw_dir, exist_ok=True)

    def download_data(self):
        print("[START] starting Smart Fallback Download...")
        downloaded_files = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Referer': 'http://insideairbnb.com/get-the-data/',
        }
        session = requests.Session()
        session.headers.update(headers)

        for city, candidates in self.CITY_CANDIDATES.items():
            print(f"\n[PROCESSING] {city} (Trying {len(candidates)} candidates)...")
            success = False
            
            for period, url in candidates:
                filename = f"{city}_{period}_listings.csv.gz"
                filepath = os.path.join(self.raw_dir, filename)
                
                # Setup: Check if already exists valid
                if os.path.exists(filepath) and os.path.getsize(filepath) > 500*1024:
                     print(f"   [FOUND] {period} already exists. Using it.")
                     downloaded_files.append((city, filepath))
                     success = True
                     break
                
                print(f"   [TRYING] {period}...")
                try:
                    time.sleep(random.uniform(3, 7)) # Polite delay
                    r = session.get(url, stream=True, timeout=90)
                    
                    if r.status_code == 200:
                        with open(filepath, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"   [SUCCESS] Downloaded {period}!")
                        downloaded_files.append((city, filepath))
                        success = True
                        break # Stop looking for older dates
                    elif r.status_code == 403:
                        print(f"   [BLOCKED] 403 Forbidden for {period}.")
                    elif r.status_code == 404:
                        print(f"   [MISSING] 404 Not Found for {period}.")
                    else:
                        print(f"   [FAIL] Status {r.status_code}")
                        
                except Exception as e:
                    print(f"   [ERROR] Exception: {e}")
            
            if not success:
                print(f"   [GAVE UP] Could not download any data for {city}.")

        return downloaded_files

    def process_and_merge(self, files):
        print("\n[MERGING] Constructing Final Dataset...")
        dfs = []
        
        for city, filepath in files:
            print(f"   Reading {city} from {os.path.basename(filepath)}...")
            try:
                df = pd.read_csv(filepath, compression='gzip', low_memory=False, dtype={'price': str}, on_bad_lines='skip')
                df['city'] = city
                dfs.append(df)
            except Exception as e:
                print(f"   Error reading {filepath}: {e}")

        if not dfs:
            print("No data to merge.")
            return

        full_df = pd.concat(dfs, ignore_index=True)
        print(f"   Merged Raw Rows: {len(full_df)}")

        # --- PRE-PROCESSING ---
        print("   Cleaning 'price' column...")
        if 'price' in full_df.columns:
            full_df['price'] = full_df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            full_df['price'] = pd.to_numeric(full_df['price'], errors='coerce')
        
        print("   Creating 'log_price' target...")
        full_df = full_df.dropna(subset=['price'])
        full_df = full_df[full_df['price'] > 0]
        full_df['log_price'] = np.log(full_df['price'])
        
        output_path = os.path.join(self.output_dir, "New_Airbnb_Data.csv")
        full_df.to_csv(output_path, index=False)
        print(f"\n[COMPLETE] Final Dataset: {len(full_df)} rows.")
        print(f"Saved to: {output_path}")

if __name__ == "__main__":
    downloader = AirbnbDataDownloader()
    files = downloader.download_data()
    if files:
        downloader.process_and_merge(files)
