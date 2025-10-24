#!/usr/bin/env python3
"""
Download script for GloVe embeddings - Lakshya Khetan
Downloads and extracts GloVe 6B 100d embeddings for the Twitter sentiment analysis project
"""

import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def main():
    """Main download and extraction function"""
    print("🚀 GloVe Embeddings Downloader - Lakshya Khetan")
    print("=" * 50)
    
    # Create glove directory if it doesn't exist
    os.makedirs('glove', exist_ok=True)
    
    # GloVe download URL
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_filename = "glove/glove.6B.zip"
    target_file = "glove/glove.6B.100d.txt"
    
    # Check if target file already exists
    if os.path.exists(target_file):
        print(f"✅ {target_file} already exists!")
        return
    
    try:
        print(f"📥 Downloading GloVe embeddings from {url}")
        download_file(url, zip_filename)
        print(f"✅ Download completed: {zip_filename}")
        
        print("📦 Extracting glove.6B.100d.txt...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Extract only the 100d file we need
            zip_ref.extract('glove.6B.100d.txt', 'glove/')
        
        print("✅ Extraction completed!")
        
        # Clean up zip file to save space
        os.remove(zip_filename)
        print("🧹 Cleaned up zip file")
        
        # Verify the file
        if os.path.exists(target_file):
            size = os.path.getsize(target_file)
            print(f"📊 GloVe embeddings ready: {target_file} ({size/1024/1024:.1f} MB)")
        else:
            print("❌ Error: Target file not found after extraction")
            
    except Exception as e:
        print(f"❌ Error downloading GloVe embeddings: {e}")
        print("💡 You can manually download from: https://nlp.stanford.edu/projects/glove/")
        print("   Extract glove.6B.100d.txt to the glove/ directory")

if __name__ == "__main__":
    main()