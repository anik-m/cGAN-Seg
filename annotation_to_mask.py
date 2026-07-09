import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image

def create_mask_from_voc(xml_path):
    """Parses a single VOC XML file and returns a PIL Image mask."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                mask[ymin:ymax, xmin:xmax] = 255
                
        return Image.fromarray(mask)
    
    except Exception as e:
        print(f"Error processing {xml_path.name}: {e}")
        return None

def batch_process_masks(input_dir, output_dir):
    """Processes all XML files in the input directory and saves masks to output directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(input_path.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in '{input_dir}'.")
        return

    print(f"Found {len(xml_files)} XML files. Starting conversion...")
    
    count = 0
    for xml_file in xml_files:
        mask_img = create_mask_from_voc(xml_file)
        
        if mask_img is not None:
            output_file_name = xml_file.stem + "_mask.png"
            mask_img.save(output_path / output_file_name)
            count += 1

    print(f"Successfully generated {count} masks in '{output_dir}'.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a folder of Pascal VOC XML annotations into binary image masks."
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input directory containing XML files."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Path to the output directory where masks will be saved."
    )
    
    args = parser.parse_args()
    
    batch_process_masks(args.input, args.output)
