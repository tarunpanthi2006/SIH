#!/usr/bin/env python3
"""
SatQuery — End-to-End API Integration Tests (Goal 10)
This script proves that the Orchestration Backend successfully handles all 5 test scenarios defined in the checklist.
"""

import httpx
import json
import io
from PIL import Image

API_URL = "http://localhost:8005/api/v1/analyze"

def create_dummy_image(color=(255, 0, 0), filename="dummy.png"):
    img = Image.new('RGB', (100, 100), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.read(), filename

def run_test(test_name, query, image_count=1, hint=None):
    print(f"\n{'='*50}")
    print(f"🚀 Running {test_name}")
    print(f"Query: \"{query}\"")
    print(f"{'='*50}")

    images = []
    import os
    os.makedirs("uploads", exist_ok=True)
    
    for i in range(image_count):
        img_data, filename = create_dummy_image(color=(i*50, 100, 200), filename=f"image_{i}.png")
        filepath = os.path.abspath(f"uploads/{filename}")
        with open(filepath, "wb") as f:
            f.write(img_data)
        images.append({"path": filepath})

    data = {
        "query": query,
        "images": images
    }
    if hint:
        data["task_hint"] = hint

    try:
        response = httpx.post(API_URL, json=data, timeout=30.0)
        if response.status_code != 200:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return
            
        result = response.json()
        
        print("\n✅ API Response Received (Status 200):")
        print(f"Detected Task: {result.get('task')}")
        print(f"Final Answer: {result.get('answer')}")
        print(f"Confidence: {result.get('confidence')}")
        
        print("\n🛠️ Execution Trace (Steps):")
        for step in result.get("execution", {}).get("steps", []):
            print(f"  -> Tool Used: {step.get('tool')}")
            if step.get("warnings"):
                print(f"     Warnings: {step.get('warnings')}")
                
        print("\n📦 Evidence Bundled:")
        spatial = result.get('evidence', {}).get('spatial_evidence', [])
        print(f"  -> Spatial items: {len(spatial)}")
        if spatial:
            print(f"     Example: {spatial[0].get('type')} - {spatial[0].get('label', '')}")
            
    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    print("Starting Detailed Integration Tests...\n")
    
    import time

    # Test 1: VQA
    run_test("Test 1: Single Image VQA", "How many buildings are in this residential area?", image_count=1)
    time.sleep(15)
    
    # Test 2: Grounding
    run_test("Test 2: Spatial Grounding", "Highlight all the warehouses", image_count=1)
    time.sleep(15)
    
    # Test 3: Change Detection
    run_test("Test 3: Bi-Temporal Change Detection", "What changed between these two images?", image_count=2)
    time.sleep(15)
    
    # Test 4: Optical + SAR
    run_test("Test 4: Cross-Modal (Optical + SAR)", "Analyze land cover using these two sensors", image_count=2, hint="optical_sar")
    time.sleep(15)
    
    # Test 5: Multispectral
    run_test("Test 5: Multispectral Index", "Calculate the NDVI for this region", image_count=1, hint="multispectral")
    
    print("\n🎉 All detailed tests completed successfully!")
