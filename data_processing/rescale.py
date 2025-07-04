import os
import argparse
from PIL import Image
import shutil
from tqdm import tqdm
import multiprocessing

#!/usr/bin/env python3
import concurrent.futures

def parse_args():
    parser = argparse.ArgumentParser(description='Rescale images and masks to a specific size')
    parser.add_argument('--input_dir', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dataset directory')
    parser.add_argument('--size', type=int, required=True, help='Target size (width=height)')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Number of parallel workers')
    return parser.parse_args()

def rescale_image(input_path, output_path, size):
    """Rescale a single image to the target size."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        img = Image.open(input_path)
        img = img.resize((size, size), Image.LANCZOS)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_folder(input_folder, output_folder, size, num_workers):
    """Process all images in a folder."""
    if not os.path.exists(input_folder):
        print(f"Warning: {input_folder} does not exist. Skipping.")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    tasks = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, rel_path)
                tasks.append((input_path, output_path, size))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda x: rescale_image(*x), tasks), total=len(tasks), desc=f"Processing {os.path.basename(input_folder)}"))

def main():
    args = parse_args()
    
    splits = ['train', 'val']
    subdirs = ['2008', '2009']
    
    for split in splits:
        for subdir in subdirs:
            input_folder = os.path.join(args.input_dir, split, subdir)
            output_folder = os.path.join(args.output_dir, split, subdir)
            process_folder(input_folder, output_folder, args.size, args.num_workers)
    
    print(f"Rescaling complete. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()