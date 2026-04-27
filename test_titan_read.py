import json
import os
from pathlib import Path
import re

# ==============================================================================
# Helper functions (from your prep script)
# ==============================================================================
def get_first(record, keys, default=None):
    for k in keys:
        if k in record:
            return record[k]
    return default

def norm_str(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple, set)):
        return " ".join(norm_str(v) for v in x)
    if isinstance(x, dict):
        return " ".join(f"{k}:{norm_str(v)}" for k, v in x.items())
    return str(x).lower()

def frame_number_from_name(name, fallback):
    nums = re.findall(r"\d+", Path(str(name)).stem)
    return int(nums[-1]) if nums else int(fallback)

# ==============================================================================
# Main Test Script
# ==============================================================================
def test_titan_dataset(titan_root_path):
    print(f"=== Testing TITAN Dataset Reader ===")
    titan_root = Path(titan_root_path)
    
    # 1. Check directories
    ann_root = titan_root / "annotations"
    img_root = titan_root / "images_anonymized"
    
    if not ann_root.exists():
        print(f"❌ Annotations folder not found at: {ann_root}")
        return
    if not img_root.exists():
        print(f"❌ Images folder not found at: {img_root}")
        return
        
    print(f"✅ Found annotations folder: {ann_root}")
    print(f"✅ Found images folder: {img_root}")

    # 2. Find the first JSON file
    ann_files = sorted(list(ann_root.glob("*.json")))
    if not ann_files:
        print("❌ No JSON files found in annotations folder.")
        return
        
    first_json = ann_files[0]
    clip_id = first_json.stem
    print(f"\n=== Reading Clip: {clip_id} ===")
    print(f"File: {first_json.name}")
    
    # 3. Check corresponding image folder
    clip_img_dir = img_root / clip_id
    if not clip_img_dir.exists():
        print(f"⚠️ Warning: Image folder for clip {clip_id} not found at {clip_img_dir}")
        print("Checking if images are nested inside an 'images' subfolder...")
        if (img_root / clip_id / "images").exists():
            clip_img_dir = img_root / clip_id / "images"
            print(f"✅ Found images in: {clip_img_dir}")
    else:
         print(f"✅ Found image folder: {clip_img_dir}")
         
    # Count images
    if clip_img_dir.exists():
        img_count = len(list(clip_img_dir.glob("*.png")))
        print(f"✅ Found {img_count} .png images for this clip.")

    # 4. Open and parse the JSON
    print("\n=== Parsing JSON Content ===")
    try:
        with open(first_json, 'r') as f:
            data = json.load(f)
            
        print(f"Data type loaded: {type(data)}")
        
        # Try to figure out the JSON structure (TITAN usually has a specific flat or nested format)
        sample_objects = []
        
        if isinstance(data, list):
             print(f"Found a list of {len(data)} frame entries.")
             if len(data) > 0:
                 # Get the first frame that has objects
                 for item in data:
                     objects = get_first(item, ["objects", "annotations", "agents", "instances", "labels"], [])
                     if objects:
                         sample_objects = objects
                         frame_idx = item.get("frame_idx", "Unknown")
                         print(f"Found objects in frame {frame_idx}")
                         break
                         
        elif isinstance(data, dict):
            if "images" in data and "annotations" in data:
                print(f"Found COCO-style format. {len(data['images'])} images, {len(data['annotations'])} annotations.")
                if len(data["annotations"]) > 0:
                    sample_objects = [data["annotations"][0]]
            else:
                 print(f"Found Dictionary format with keys: {list(data.keys())[:5]}...")
                 # Grab the first value that looks like a list of objects
                 for k, v in data.items():
                     if isinstance(v, list) and len(v) > 0:
                         sample_objects = v
                         print(f"Found objects under key/frame: {k}")
                         break

        # 5. Print a sample object to verify fields
        if sample_objects:
            print("\n=== Sample Pedestrian Object ===")
            obj = sample_objects[0]
            print(json.dumps(obj, indent=2))
            
            # Check specific fields PIPNet needs
            bbox = get_first(obj, ["bbox", "box", "bb", "bounds"])
            tid = get_first(obj, ["track_id", "trackId", "tid", "id", "object_id", "agent_id"])
            cls = norm_str(get_first(obj, ["category", "category_name", "class", "label", "type"]))
            
            print("\n=== Extraction Check ===")
            print(f"Bounding Box format: {bbox}")
            print(f"Track ID: {tid}")
            print(f"Class/Category: {cls}")
            
            # Check for action labels
            action_text = " ".join(norm_str(obj.get(k, "")) for k in [
                "action", "actions", "label", "labels", "behavior", "attributes"
            ])
            print(f"Extracted Action Text: '{action_text}'")
            
        else:
            print("❌ Could not find any pedestrian objects in the parsed data structure.")

    except Exception as e:
        print(f"❌ Error reading JSON: {e}")

if __name__ == "__main__":
    # Change this path to where your TITAN dataset actually lives!
    TITAN_PATH = "/data/TITAN/titan_data/dataset"
    test_titan_dataset(TITAN_PATH)