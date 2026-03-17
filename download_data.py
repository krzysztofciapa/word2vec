import os
import zipfile
import urllib.request

def download_text8(data_dir="data"):

    os.makedirs(data_dir, exist_ok=True)
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = os.path.join(data_dir, "text8.zip")
    extract_path = os.path.join(data_dir, "text8")

    if os.path.exists(extract_path):
        print(f"Dataset already exists at: {extract_path}")
        return extract_path

    print(f"Downloading text8 corpus from {url}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int((downloaded / total_size) * 100))
        print(f"\rDownloading: {percent}%", end="")

    urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
    print("\nDownload complete. Extracting...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_path)
    print(f"Extraction complete. Corpus ready at: {extract_path}")
    return extract_path

if __name__ == "__main__":
    download_text8()