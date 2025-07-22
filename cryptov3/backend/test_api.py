import requests
import os
import sys
import json
from pathlib import Path
import hashlib

# Configuration
API_URL = 'http://localhost:8000'  # Change if your API is hosted elsewhere

def test_predict_endpoint(image_path, timeframe='1h'):
    """
    Test the /api/predict endpoint with a local image file
    
    Args:
        image_path: Path to the test image
        timeframe: Chart timeframe (15m, 1h, 4h, 1d, 7d)
    """
    print(f"\n--- Testing /api/predict with {image_path}, timeframe: {timeframe} ---")
    
    # Calculate image hash for verification
    with open(image_path, 'rb') as f:
        image_content = f.read()
        image_hash = hashlib.md5(image_content).hexdigest()
        print(f"Calculated image hash: {image_hash}")
    
    # Create form data with image and timeframe
    files = {
        'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/png')
    }
    data = {
        'timeframe': timeframe
    }
    
    # Send request to API
    try:
        print(f"Sending request to {API_URL}/api/predict...")
        response = requests.post(f"{API_URL}/api/predict", files=files, data=data)
        response.raise_for_status()
        
        # Process response
        result = response.json()
        
        # Print the full response
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
        
        # Check if image_hash is in response
        if 'image_hash' in result:
            print(f"\nImage hash in response: {result['image_hash']}")
            print(f"Does it match our calculated hash? {'Yes' if result['image_hash'] == image_hash else 'No'}")
        else:
            print("\nWARNING: No image_hash in response")
            
        # Print key results
        print("\nKey Results:")
        print(f"Prediction: {result.get('prediction', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        if 'patterns' in result:
            print(f"Patterns: {result.get('patterns', [])}")
        if 'detected_patterns' in result:
            print(f"Detected Patterns: {result.get('detected_patterns', [])}")
        
        # Test consistency by sending the same request again
        print("\nTesting consistency - sending the same request again...")
        response2 = requests.post(f"{API_URL}/api/predict", files=files, data=data)
        result2 = response2.json()
        
        # Check if results are consistent
        is_consistent = (
            result.get('prediction') == result2.get('prediction') and
            result.get('confidence') == result2.get('confidence') and
            result.get('image_hash') == result2.get('image_hash')
        )
        
        print(f"Results consistent across requests: {'Yes' if is_consistent else 'No'}")
        
        return result
        
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    finally:
        # Close the file handle
        files['image'][1].close()

def test_imgNewModel_endpoint(image_path, timeframe='1h'):
    """
    Test the /api/imgNewModel endpoint with a local image file
    
    Args:
        image_path: Path to the test image
        timeframe: Chart timeframe (15m, 1h, 4h, 1d, 7d)
    """
    print(f"\n--- Testing /api/imgNewModel with {image_path}, timeframe: {timeframe} ---")
    
    # First check if the endpoint exists
    try:
        print(f"Checking available endpoints at {API_URL}...")
        response = requests.get(f"{API_URL}/")
        print(f"API root response: {response.status_code}")
        if response.status_code == 200:
            print(f"API root content: {response.text[:500]}...")
    except requests.RequestException as e:
        print(f"Error checking API root: {e}")
    
    # Calculate image hash for verification
    with open(image_path, 'rb') as f:
        image_content = f.read()
        image_hash = hashlib.md5(image_content).hexdigest()
        print(f"Calculated image hash: {image_hash}")
    
    # Create form data with image and timeframe
    files = {
        'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/png')
    }
    data = {
        'timeframe': timeframe
    }
    
    # Send request to API
    try:
        print(f"Sending request to {API_URL}/api/imgNewModel...")
        response = requests.post(f"{API_URL}/api/imgNewModel", files=files, data=data)
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()
        
        # Process response
        result = response.json()
        
        # Print the full response
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
        
        # Check if image_hash is in response
        if 'image_hash' in result:
            print(f"\nImage hash in response: {result['image_hash']}")
            print(f"Does it match our calculated hash? {'Yes' if result['image_hash'] == image_hash else 'No'}")
        else:
            print("\nWARNING: No image_hash in response")
        
        return result
        
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        # Try to get the response content even if there was an error
        try:
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error response content: {e.response.text}")
        except:
            pass
        return None
    finally:
        # Close the file handle
        files['image'][1].close()

def find_test_images(search_dir=None):
    """
    Find test images in the project
    """
    if search_dir is None:
        # If no dir provided, search in current dir and a few common locations
        search_dir = os.getcwd()
        common_dirs = [
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
                # Look for PNG and JPG files
                images = list(Path(d).glob('*.png')) + list(Path(d).glob('*.jpg'))
                if images:
                    return [str(img) for img in images]
    
    # If still no images found, look for any in current directory
    images = list(Path(search_dir).glob('**/*.png')) + list(Path(search_dir).glob('**/*.jpg'))
    return [str(img) for img in images[:5]]  # Return first 5 images max

if __name__ == "__main__":
    # Check if specific image path is provided
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        timeframe = sys.argv[2] if len(sys.argv) > 2 else '1h'
        
        if not os.path.exists(test_image):
            print(f"Error: Image file not found: {test_image}")
            sys.exit(1)
    else:
        # Find test images automatically
        test_images = find_test_images()
        if not test_images:
            print("Error: No test images found. Please provide an image path.")
            sys.exit(1)
        
        test_image = test_images[0]
        timeframe = '1h'
        
        print(f"Found {len(test_images)} test images.")
        print(f"Using first image: {test_image}")
    
    # Test both endpoints
    predict_result = test_predict_endpoint(test_image, timeframe)
    imgNewModel_result = test_imgNewModel_endpoint(test_image, timeframe)
    
    # Compare results from both endpoints
    if predict_result and imgNewModel_result:
        print("\n--- Comparing results from both endpoints ---")
        predict_prediction = predict_result.get('prediction', 'N/A')
        newModel_prediction = imgNewModel_result.get('prediction', 'N/A') if imgNewModel_result.get('isChart', False) else 'N/A'
        
        print(f"predict endpoint prediction: {predict_prediction}")
        print(f"imgNewModel endpoint prediction: {newModel_prediction}")
        print(f"Same prediction: {'Yes' if predict_prediction == newModel_prediction else 'No'}")
