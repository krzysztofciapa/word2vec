import os
import zipfile
import urllib.request


DATA_DIR = "data"

#training dataset
def download_text8():
    
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = os.path.join(DATA_DIR, "text8.zip")
    extract_path = os.path.join(DATA_DIR, "text8")

    if os.path.exists(extract_path):
        print(f"Dataset already exists at: {extract_path}")
        return extract_path

    print(f"Downloading text8 corpus from {url}")
    urllib.request.urlretrieve(url, zip_path, reporthook=_report_progress)
    print("\nDownload complete. Extracting...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    os.remove(zip_path)
    print(f"Corpus ready at: {extract_path}")
    return extract_path

#benchmarks
def download_analogy_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
    output_path = os.path.join(DATA_DIR, "questions-words.txt")

    if os.path.exists(output_path):
        print(f"Analogy dataset already exists at: {output_path}")
        return output_path

    print(f"Downloading analogy dataset from {url}")
    urllib.request.urlretrieve(url, output_path, reporthook=_report_progress)
    print(f"\nAnalogy dataset ready at: {output_path}")
    return output_path


def download_wordsim353():
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/exact/EN-WS-353-ALL.txt"
    output_path = os.path.join(DATA_DIR, "wordsim353.tsv")

    if os.path.exists(output_path):
        print(f"WordSim-353 already exists at: {output_path}")
        return output_path

    print(f"Downloading WordSim-353 from {url}")
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=_report_progress)
        print(f"\nWordSim-353 ready at: {output_path}")
    except Exception as e:
        print(f"\nFailed to download WordSim-353: {e}")
    return output_path


def _report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, int((downloaded / total_size) * 100))
        print(f"\rDownloading: {percent}%", end="")
    else:
        print(f"\rDownloaded: {downloaded / 1024:.0f} KB", end="")


if __name__ == "__main__":
    download_text8()
    download_analogy_dataset()
    download_wordsim353()
    print("\nAll datasets ready.")