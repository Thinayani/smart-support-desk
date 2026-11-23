"""Download the Bitext dataset to data/bitext.csv (needs internet access to huggingface.co).

Run:  python ml/download_data.py
Alternative: download the CSV from the dataset's "Files and versions" tab and save it as data/bitext.csv.
"""
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "bitext.csv"

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    ds.to_csv(str(OUT))
    print(f"saved {len(ds)} rows to {OUT}")
