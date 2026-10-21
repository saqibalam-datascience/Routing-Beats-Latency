"""
Helper Script to Download and Extract the Imagenette Vision Benchmark.
"""
import os
import sys
import tarfile
import urllib.request
from tqdm import tqdm

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_imagenette(target_dir: str = None):
    if target_dir is None:
        target_dir = os.path.join(os.path.dirname(__file__), "..", "data", "imagenette")
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    archive_path = os.path.join(target_dir, "imagenette2-320.tgz")
    extract_folder = os.path.join(target_dir, "imagenette2-320")

    if os.path.exists(os.path.join(target_dir, "train")) and os.path.exists(os.path.join(target_dir, "val")):
        print(f"Imagenette already found and prepared at: {target_dir}")
        return

    if not os.path.exists(archive_path):
        print(f"Downloading Imagenette (320px) from {IMAGENETTE_URL}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Imagenette") as t:
            urllib.request.urlretrieve(IMAGENETTE_URL, filename=archive_path, reporthook=t.update_to)
        print("Download complete.")

    print(f"Extracting archive to {target_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    # Move train/val directly under target_dir if extracted to subfolder
    if os.path.exists(extract_folder):
        for sub in ["train", "val"]:
            src = os.path.join(extract_folder, sub)
            dst = os.path.join(target_dir, sub)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

    print(f"Imagenette ready at: {target_dir}")

if __name__ == "__main__":
    download_imagenette()
