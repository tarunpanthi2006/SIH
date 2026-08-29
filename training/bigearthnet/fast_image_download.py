import os
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def download_fast():
    rgb_dir = Path("datasets/bigearthnet/rgb")
    rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the list of 80,000 patches we actually need
    instructions_file = Path("datasets/bigearthnet/processed/bigearthnet_instructions.json")
    if not instructions_file.exists():
        print(f"Error: {instructions_file} not found.")
        print("Please run the prepare and convert commands first!")
        return
        
    print("Loading instruction list to find required images...")
    with open(instructions_file, "r") as f:
        data = json.load(f)
        
    needed_patches = set()
    for item in data:
        patch_id = item["metadata"]["patch_id"]
        needed_patches.add(patch_id)
        
    print(f"Total unique images required for training: {len(needed_patches)}")
    
    # Filter out ones we might have already downloaded if the script restarted
    existing = set([f.stem for f in rgb_dir.glob("*.png")])
    needed_patches = needed_patches - existing
    print(f"Remaining images to download: {len(needed_patches)}")
    
    if len(needed_patches) == 0:
        print("All images are already downloaded!")
        return

    print("\nStreaming images from HuggingFace CDN (Lightning Fast!)...")
    print("This skips Zenodo completely and only downloads the images you need.")
    
    # decode=False gives us raw bytes and the original filename instantly!
    ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", "s2-rgb", split="train", streaming=True, token="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").cast_column("img", datasets.Image(decode=False))
    
    saved_count = 0
    for item in ds:
        # With decode=False, item["img"] is a dictionary: {'path': 'filename.png', 'bytes': b'...'}
        img_data = item["img"]
        
        # Get the patch_id from the original file path stored in the dataset
        original_filename = img_data["path"]
        patch_id = Path(original_filename).stem
            
        # If this image is one of our required 80k, save it!
        if patch_id in needed_patches:
            out_path = rgb_dir / f"{patch_id}.png"
            
            # Save the raw bytes directly to disk (this is 10x faster than PIL Image.save!)
            with open(out_path, "wb") as f:
                f.write(img_data["bytes"])
            
            needed_patches.remove(patch_id)
            saved_count += 1
            
            if saved_count % 1000 == 0:
                print(f"Saved {saved_count} images... {len(needed_patches)} remaining")
                
            # Stop streaming if we found all our images!
            if len(needed_patches) == 0:
                break
                
    print(f"\nDone! Successfully downloaded {saved_count} images directly to the rgb folder.")
    print("You are now completely ready to train the model!")

if __name__ == "__main__":
    download_fast()
