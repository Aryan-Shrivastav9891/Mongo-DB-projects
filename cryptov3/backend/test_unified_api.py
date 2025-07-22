import requests
import os
import sys
import json
from pathlib import Path
import hashlib
import time

# Configuration
API_URL = 'http://localhost:8000'  # Change if your API is hosted elsewhere

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            print(f"API health endpoint responded with: {response.json()}")
            return True
        else:
            print(f"API health endpoint responded with status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")
        return False

def list_api_endpoints():
    """List all available endpoints"""
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            print("Available endpoints from root response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error getting API endpoints: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")

def find_test_images(search_dir=None):
    """Find test images in the project"""
    if search_dir is None:
        # If no dir provided, search in current dir and a few common locations
        search_dir = os.getcwd()
        common_dirs = [
            os.path.join(search_dir, "test_images"),
            os.path.join(os.path.dirname(search_dir), 'test_images'),
            os.path.join(os.path.dirname(search_dir), 'tests', 'images'),
            os.path.join(os.path.dirname(search_dir), 'public', 'images'),
            os.path.join(os.path.dirname(search_dir), '..', 'public', 'images'),
        ]
        
        # Add search dir to the list
        common_dirs.insert(0, search_dir)
        
        # Look for images in common locations
        for d in common_dirs:
            if os.path.exists(d):
                print(f"Searching for images in: {d}")
                # Look for PNG and JPG files
                images = list(Path(d).glob('*.png')) + list(Path(d).glob('*.jpg'))
                if images:
                    return [str(img) for img in images]
    
    # If still no images found, look for any in current directory
    images = list(Path(search_dir).glob('**/*.png')) + list(Path(search_dir).glob('**/*.jpg'))
    return [str(img) for img in images[:5]]  # Return first 5 images max

def test_endpoint(endpoint_path, image_path, timeframe='1h'):
    """Test an API endpoint with an image file"""
    print(f"\n--- Testing {endpoint_path} with {os.path.basename(image_path)}, timeframe: {timeframe} ---")
    
    # Calculate image hash for verification
    with open(image_path, 'rb') as f:
        image_content = f.read()
        image_hash = hashlib.md5(image_content).hexdigest()
        print(f"Image hash: {image_hash}")
    
    # Create form data with image and timeframe
    files = {
        'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/png')
    }
    data = {
        'timeframe': timeframe
    }
    
    # Send request to API
    try:
        url = f"{API_URL}{endpoint_path}"
        print(f"Sending request to {url}")
        response = requests.post(url, files=files, data=data)
        print(f"Response status code: {response.status_code}")
        
        # Try to parse response as JSON
        try:
            result = response.json()
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            
            # Check if image_hash is in response
            if 'image_hash' in result:
                print(f"\nImage hash in response: {result['image_hash']}")
                print(f"Does it match our calculated hash? {'Yes' if result['image_hash'] == image_hash else 'No'}")
            
            return result
        except ValueError:
            print("\nResponse is not valid JSON:")
            print(response.text[:500])  # Show first 500 chars
            return None
            
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    finally:
        # Close the file handle
        files['image'][1].close()

def main():
    print("=== Crypto API Test Suite ===")
    
    # First check if API is healthy
    if not check_api_health():
        print("\nThe API does not appear to be running. Please start the API server first.")
        return
    
    # List available endpoints
    list_api_endpoints()
    
    # Find test images
    test_images = find_test_images()
    if not test_images:
        print("\nNo test images found. Creating a sample image...")
        
        # Create a simple test image
        try:
            import numpy as np
            from PIL import Image
            
            # Create a simple gradient image
            width, height = 400, 300
            img = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    img[i, j] = [i % 256, j % 256, (i + j) % 256]
            
            # Save the image
            test_image_path = os.path.join(os.getcwd(), "test_image.png")
            Image.fromarray(img).save(test_image_path)
            print(f"Created test image at: {test_image_path}")
            
            test_images = [test_image_path]
        except ImportError:
            print("Could not create test image. Please provide an image path.")
            return
    
    print(f"\nFound {len(test_images)} test images:")
    for i, img in enumerate(test_images):
        print(f"{i+1}. {img}")
    
    # Use the first test image
    test_image = test_images[0]
    print(f"\nUsing test image: {test_image}")
    
    # Test endpoints
    endpoints = [
        "/api/predict",
        "/api/imgNewModel"
    ]
    
    results = {}
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        result = test_endpoint(endpoint, test_image, "1h")
        results[endpoint] = result
        
        # Give the server a moment to breathe
        time.sleep(1)
    
    # Print summary
    print("\n=== Test Summary ===")
    for endpoint, result in results.items():
        status = "✓ Success" if result is not None else "✗ Failed"
        prediction = result.get("prediction", "N/A") if result else "N/A"
        print(f"{endpoint}: {status} (Prediction: {prediction})")

if __name__ == "__main__":
    main()
