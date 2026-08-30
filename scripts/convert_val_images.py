import json, os, numpy as np
from pathlib import Path
from PIL import Image

try:
    import rasterio
    USE_RASTERIO = True
except ImportError:
    USE_RASTERIO = False
    print("Warning: rasterio not found, using PIL to read TIFs")

val_path = Path("datasets/bigearthnet/processed/val_small.json")
rgb_dir = Path("datasets/bigearthnet/rgb")

if not val_path.exists():
    print(f"Error: {val_path} not found!")
    exit(1)

with open(val_path) as f:
    data = json.load(f)

# Get all unique patch IDs from the val set
needed = set([Path(s.get("image")).stem for s in data if s.get("image")])
print(f"Need to generate RGB PNG composites for {len(needed)} val patches...")

created = 0
missing = 0

for patch_id in needed:
    out = rgb_dir / f"{patch_id}.png"
    if out.exists():
        continue
    
    # We need Red (B04), Green (B03), Blue (B02)
    bands = {}
    for b in ["B04", "B03", "B02"]:
        p = rgb_dir / f"{patch_id}_{b}.tif"
        if p.exists():
            bands[b] = p
        
    if len(bands) == 3:
        try:
            if USE_RASTERIO:
                arrs = [rasterio.open(bands[b]).read(1) for b in ["B04", "B03", "B02"]]
            else:
                arrs = [np.array(Image.open(bands[b])) for b in ["B04", "B03", "B02"]]
            
            # Stack into RGB array
            rgb = np.stack(arrs, axis=-1).astype(np.float32)
            
            # Normalize for visualization (clip 2nd/98th percentile for contrast)
            for c in range(3):
                p2 = np.percentile(rgb[:,:,c], 2)
                p98 = np.percentile(rgb[:,:,c], 98)
                if p98 > p2:
                    rgb[:,:,c] = np.clip((rgb[:,:,c] - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    rgb[:,:,c] = 0
                    
            # Save as PNG
            Image.fromarray(rgb.astype(np.uint8)).save(out)
            created += 1
            if created % 50 == 0:
                print(f"  Created {created} PNG composites so far...")
        except Exception as e:
            print(f"Failed to create composite for {patch_id}: {e}")
    else:
        missing += 1

print(f"\n✅ Created {created} new PNG composites in {rgb_dir}!")
if missing > 0:
    print(f"⚠️ {missing} patches were missing some raw TIF bands.")
