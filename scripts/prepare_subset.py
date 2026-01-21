import argparse
import json
import os
import random
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_image(args):
    """
    Download a single image.
    args: (image_filename, save_path)
    """
    filename, save_dir = args
    
    # Try to handle different filename formats to map to COCO 2017
    # COCO 2017 images are usually named like 000000123456.jpg
    # LLaVA 150k often uses COCO_train2014_000000123456.jpg
    
    url_filename = filename
    if "COCO_train2014_" in filename:
        # Extract the ID and format to 2017 standard if needed, 
        # OR just try to download the file as is if it exists on 2014 server.
        # But user specifically asked for 2017 images. 
        # COCO IDs are consistent across splits usually.
        # Let's try to extract the ID.
        image_id_str = filename.split("_")[-1] # 000000123456.jpg
        url_filename = image_id_str
    
    # COCO 2017 Train URL
    url = f"http://images.cocodataset.org/train2017/{url_filename}"
    
    save_path = save_dir / filename # Keep original filename to match JSON
    
    if save_path.exists():
        return True

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            # Fallback: Maybe it's not in train2017, try val2017
            url_val = f"http://images.cocodataset.org/val2017/{url_filename}"
            response_val = requests.get(url_val, timeout=10)
            if response_val.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response_val.content)
                return True
            return False
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare a random subset of data for rapid testing.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory containing source JSONs")
    parser.add_argument("--output_dir", type=str, default="./data/subset", help="Directory to save subset data")
    parser.add_argument("--num_sample", type=int, default=512, help="Number of samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=8, help="Number of download threads")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load LLaVA Instruct 150k
    llava_json_path = data_root / "llava_instruct_150k.json"
    if not llava_json_path.exists():
        print(f"Error: {llava_json_path} not found.")
        return
        
    print(f"Loading {llava_json_path}...")
    with open(llava_json_path, 'r') as f:
        llava_data = json.load(f)
        
    # 2. Sample
    if len(llava_data) > args.num_sample:
        print(f"Sampling {args.num_sample} items from {len(llava_data)} total...")
        llava_subset = random.sample(llava_data, args.num_sample)
    else:
        llava_subset = llava_data
        
    # 3. Extract Image List
    subset_images = set()
    for item in llava_subset:
        if 'image' in item:
            subset_images.add(item['image'])
            
    print(f"Found {len(subset_images)} unique images to download.")
    
    # 4. Download Images
    download_tasks = [(img_name, images_dir) for img_name in subset_images]
    
    print("Downloading images...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks)))
        success_count = sum(results)
        
    print(f"Successfully downloaded {success_count}/{len(download_tasks)} images.")
    
    # 5. Save LLaVA Subset
    llava_subset_path = output_dir / "llava_instruct_150k.json"
    with open(llava_subset_path, 'w') as f:
        json.dump(llava_subset, f, indent=2)
    print(f"Saved LLaVA subset to {llava_subset_path}")
    
    # 6. Process Captions JSON (if exists)
    captions_path = data_root / "captions_train2017.json"
    if captions_path.exists():
        print(f"Processing {captions_path}...")
        with open(captions_path, 'r') as f:
            coco_data = json.load(f)
            
        # We need to filter based on the images we actually kept
        # COCO filenames in annotations usually don't have the path, just the name
        # We need to match the 'file_name' in coco_data['images'] with our subset_images
        
        # Build mapping of image filenames we have
        # Note: LLaVA might use 'COCO_train2014_...' but COCO2017 uses '00000...'
        # We need to be careful with matching.
        
        # Let's normalize our subset_images to just the ID part for matching against COCO 2017
        subset_ids = set()
        for img_name in subset_images:
             # Extract numeric ID: COCO_train2014_000000123456.jpg -> 123456
             # or 000000123456.jpg -> 123456
            try:
                img_id = int(img_name.split('_')[-1].split('.')[0])
                subset_ids.add(img_id)
            except:
                pass

        new_images = []
        kept_image_ids = set()
        
        for img in coco_data['images']:
            if img['id'] in subset_ids:
                new_images.append(img)
                kept_image_ids.add(img['id'])
                
        new_annotations = []
        for ann in coco_data['annotations']:
            if ann['image_id'] in kept_image_ids:
                new_annotations.append(ann)
                
        coco_subset = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "images": new_images,
            "annotations": new_annotations
        }
        
        captions_subset_path = output_dir / "captions_subset.json"
        with open(captions_subset_path, 'w') as f:
            json.dump(coco_subset, f, indent=2)
            
        print(f"Saved COCO Captions subset to {captions_subset_path} (Images: {len(new_images)}, Annotations: {len(new_annotations)})")
    else:
        print(f"Warning: {captions_path} not found, skipping captions subset generation.")

if __name__ == "__main__":
    main()
