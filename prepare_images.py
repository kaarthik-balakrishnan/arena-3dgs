#!/usr/bin/env python3
"""
Preprocess images for Gaussian Splatting on Google Colab.
Resizes images to reduce file size while maintaining quality for SfM.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import shutil

class ImagePreprocessor:
    def __init__(self, input_folder, output_folder, max_width=1920, quality=90):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.max_width = max_width
        self.quality = quality
        
    def resize_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not load {img_path.name}")
            return None
            
        height, width = img.shape[:2]
        
        if width > self.max_width:
            scale = self.max_width / width
            new_width = self.max_width
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"  Resized {img_path.name}: {width}x{height} -> {new_width}x{new_height}")
        
        return img
    
    def process(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(self.input_folder.glob('*.jpg')) + \
                      sorted(self.input_folder.glob('*.jpeg')) + \
                      sorted(self.input_folder.glob('*.png'))
        
        print(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            img = self.resize_image(img_path)
            if img is not None:
                output_path = self.output_folder / img_path.name
                cv2.imwrite(str(output_path), img, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        
        print(f"\nProcessed images saved to: {self.output_folder}")
        
        original_size = sum(f.stat().st_size for f in image_files)
        processed_size = sum(f.stat().st_size for f in self.output_folder.glob('*.*'))
        
        print(f"Original size: {original_size / 1024 / 1024:.1f} MB")
        print(f"Processed size: {processed_size / 1024 / 1024:.1f} MB")
        print(f"Compression ratio: {original_size / processed_size:.1f}x")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess images for Gaussian Splatting')
    parser.add_argument('--input', '-i', default='splat-files', help='Input folder')
    parser.add_argument('--output', '-o', default='splat-files-processed', help='Output folder')
    parser.add_argument('--width', '-w', type=int, default=1920, help='Max width (default: 1920)')
    parser.add_argument('--quality', '-q', type=int, default=90, help='JPEG quality 0-100 (default: 90)')
    
    args = parser.parse_args()
    
    preprocessor = ImagePreprocessor(args.input, args.output, args.width, args.quality)
    preprocessor.process()
