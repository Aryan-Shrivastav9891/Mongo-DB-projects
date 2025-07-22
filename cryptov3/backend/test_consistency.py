"""
Test script to verify consistency of predictions for the same image.

This script can test both:
1. Direct use of the prediction service (in-process)
2. API endpoint consistency (HTTP requests)
"""
import os
import hashlib
import numpy as np
import requests
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Try to import the predictor - may fail if dependencies are missing
try:
    from services.prediction_service import CandlestickPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("Warning: Could not import CandlestickPredictor. API tests will still work.")

# API URL for HTTP tests
API_URL = 'http://localhost:8000'

def calculate_image_hash(image_path: str) -> str:
    """Calculate MD5 hash of an image file"""
    with open(image_path, 'rb') as f:
        image_content = f.read()
        return hashlib.md5(image_content).hexdigest()

def test_direct_prediction_consistency(test_image: str, timeframe: str = "1h", iterations: int = 5) -> bool:
    """
    Test if direct predictor calls produce consistent results for the same image.
    
    Args:
        test_image: Path to the test image
        timeframe: Chart timeframe to use
        iterations: Number of prediction iterations
        
    Returns:
        True if predictions are consistent, False otherwise
    """
    if not PREDICTOR_AVAILABLE:
        print("Cannot run direct prediction test: CandlestickPredictor not available")
        return False
        
    print(f"\n=== Testing DIRECT prediction consistency ===")
    print(f"Image: {test_image}")
    print(f"Timeframe: {timeframe}")
    print(f"Iterations: {iterations}")
    
    # Initialize predictor
    predictor = CandlestickPredictor()
    
    if not os.path.exists(test_image):
        print(f"Test image not found at {test_image}")
        return False
    
    # Calculate image hash
    image_hash = calculate_image_hash(test_image)
    print(f"Image hash: {image_hash}")
    
    # Make predictions multiple times with hash
    results_with_hash = []
    print("\nRunning predictions WITH hash (should be consistent)...")
    
    for i in range(iterations):
        result = predictor.predict(test_image, timeframe, image_hash)
        results_with_hash.append(result)
        print(f"Run {i+1}: {result['prediction']} (Confidence: {result.get('confidence', 'N/A')})")
        patterns_key = 'patterns' if 'patterns' in result else 'detected_patterns'
        print(f"  Patterns: {', '.join(result.get(patterns_key, []))}")
    
    # Check if all predictions with hash are consistent
    if results_with_hash:
        first_prediction = results_with_hash[0]['prediction']
        first_confidence = results_with_hash[0].get('confidence', 0)
        
        patterns_key = 'patterns' if 'patterns' in results_with_hash[0] else 'detected_patterns'
        first_patterns = sorted(results_with_hash[0].get(patterns_key, []))
        
        all_same = True
        for i, result in enumerate(results_with_hash[1:], 1):
            curr_patterns_key = 'patterns' if 'patterns' in result else 'detected_patterns'
            if (result['prediction'] != first_prediction or 
                result.get('confidence', 0) != first_confidence or 
                sorted(result.get(curr_patterns_key, [])) != first_patterns):
                all_same = False
                print(f"\nInconsistency detected in run {i+1}!")
                break
        
        if all_same:
            print("\n✅ SUCCESS: All predictions are consistent when using the image hash.")
        else:
            print("\n❌ FAILURE: Predictions were inconsistent even with image hash.")
            return False
    
    # Now test without using hash to demonstrate the difference
    print("\nRunning predictions WITHOUT hash (should be random)...")
    np.random.seed(None)  # Reset the random seed
    
    results_no_hash = []
    for i in range(iterations):
        result = predictor.predict(test_image, timeframe)  # No hash provided
        results_no_hash.append(result)
        print(f"Run {i+1}: {result['prediction']} (Confidence: {result.get('confidence', 'N/A')})")
        patterns_key = 'patterns' if 'patterns' in result else 'detected_patterns'
        print(f"  Patterns: {', '.join(result.get(patterns_key, []))}")
        
    # Check if predictions without hash are different (as expected)
    if results_no_hash:
        predictions = [r['prediction'] for r in results_no_hash]
        unique_predictions = set(predictions)
        if len(unique_predictions) > 1:
            print("\n✅ EXPECTED: Without hash, predictions vary as expected.")
            return True
        else:
            print("\n⚠️ UNEXPECTED: Even without hash, all predictions were the same.")
            # This is not necessarily a failure, but unexpected
            return True
    
    return True

def test_api_consistency(test_image: str, endpoint: str = "/api/predict", 
                        timeframe: str = "1h", iterations: int = 5) -> bool:
    """
    Test if API endpoint gives consistent results for the same image
    
    Args:
        test_image: Path to the test image
        endpoint: API endpoint to test
        timeframe: Chart timeframe
        iterations: Number of test iterations
        
    Returns:
        True if results are consistent, False otherwise
    """
    print(f"\n=== Testing API consistency for {endpoint} ===")
    print(f"Image: {test_image}")
    print(f"Timeframe: {timeframe}")
    print(f"Iterations: {iterations}")
    
    # Calculate image hash for verification
    image_hash = calculate_image_hash(test_image)
    print(f"Image hash: {image_hash}")
    
    results = []
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Create form data with image and timeframe
        files = {
            'image': (os.path.basename(test_image), open(test_image, 'rb'), 'image/png')
        }
        data = {
            'timeframe': timeframe
        }
        
        # Send request to API
        try:
            url = f"{API_URL}{endpoint}"
            print(f"Sending request to {url}")
            response = requests.post(url, files=files, data=data)
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {result.get('prediction', 'N/A')}, " +
                      f"Confidence: {result.get('confidence', 'N/A')}")
                
                # Check image hash in response
                if 'image_hash' in result:
                    print(f"Response image hash: {result['image_hash']}")
                    if result['image_hash'] != image_hash:
                        print("⚠️ WARNING: Image hash in response doesn't match calculated hash")
                
                results.append(result)
            else:
                print(f"Error response: {response.text}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            # Close the file handle
            files['image'][1].close()
        
        # Wait a bit between requests
        time.sleep(1)
    
    # Check if all results are consistent
    if len(results) < 2:
        print("\nNot enough successful responses to compare consistency")
        return False
    
    print("\n=== API Consistency Analysis ===")
    
    # Check prediction consistency
    predictions = [r.get('prediction') for r in results if 'prediction' in r]
    unique_predictions = set(predictions)
    
    print(f"Total responses: {len(results)}")
    print(f"Unique predictions: {len(unique_predictions)}")
    
    if len(unique_predictions) == 1:
        print(f"✅ PREDICTION CONSISTENT: All responses returned '{predictions[0]}'")
        is_prediction_consistent = True
    else:
        print("❌ PREDICTION INCONSISTENT: Responses returned different predictions:")
        for pred in unique_predictions:
            count = predictions.count(pred)
            print(f"  - '{pred}': {count} times ({count/len(predictions)*100:.1f}%)")
        is_prediction_consistent = False
    
    # Check confidence consistency
    confidences = [r.get('confidence') for r in results if 'confidence' in r]
    unique_confidences = set(confidences)
    
    if len(unique_confidences) == 1:
        print(f"✅ CONFIDENCE CONSISTENT: All responses returned {confidences[0]}")
        is_confidence_consistent = True
    else:
        print("❌ CONFIDENCE INCONSISTENT: Responses returned different confidence values:")
        for conf in sorted(unique_confidences):
            count = confidences.count(conf)
            print(f"  - {conf}: {count} times ({count/len(confidences)*100:.1f}%)")
        is_confidence_consistent = False
    
    # Overall consistency
    if is_prediction_consistent and is_confidence_consistent:
        print("\n✅ OVERALL: The API gives CONSISTENT results for the same image")
        return True
    else:
        print("\n❌ OVERALL: The API gives INCONSISTENT results for the same image")
        return False

def find_test_images(search_dir=None):
    """Find test images in the project"""
    if search_dir is None:
        search_dir = os.getcwd()
    
    # Look for test images in common locations
    test_dirs = [
        search_dir,
        os.path.join(search_dir, "test_images"),
        os.path.join(os.path.dirname(search_dir), 'test_images'),
        os.path.join(os.path.dirname(search_dir), 'tests', 'images'),
    ]
    
    for directory in test_dirs:
        if os.path.exists(directory):
            images = list(Path(directory).glob('*.png')) + list(Path(directory).glob('*.jpg'))
            if images:
                return [str(img) for img in images]
    
    # If still no images found, look for any in current directory
    images = list(Path(search_dir).glob('**/*.png')) + list(Path(search_dir).glob('**/*.jpg'))
    return [str(img) for img in images[:5]]  # Return first 5 images max

def main():
    parser = argparse.ArgumentParser(description='Test API and predictor consistency')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--endpoint', default='/api/predict', help='API endpoint to test')
    parser.add_argument('--timeframe', default='1h', help='Chart timeframe')
    parser.add_argument('--iterations', type=int, default=5, help='Number of test iterations')
    parser.add_argument('--mode', choices=['api', 'direct', 'both'], default='both', 
                        help='Test mode: api, direct, or both')
    args = parser.parse_args()
    
    # If no image path provided, find test images
    if args.image:
        test_image = args.image
    else:
        # Look for test images
        test_images = find_test_images()
        
        if not test_images:
            print("No test images found. Please provide an image path with --image")
            return
        
        test_image = test_images[0]
        print(f"Using found image: {test_image}")
    
    # Run consistency tests
    success = True
    
    if args.mode in ['direct', 'both'] and PREDICTOR_AVAILABLE:
        direct_success = test_direct_prediction_consistency(
            test_image, args.timeframe, args.iterations)
        success = success and direct_success
    
    if args.mode in ['api', 'both']:
        api_success = test_api_consistency(
            test_image, args.endpoint, args.timeframe, args.iterations)
        success = success and api_success
    
    # Final summary
    print("\n=== Final Test Summary ===")
    if success:
        print("✅ All consistency tests PASSED!")
    else:
        print("❌ Some consistency tests FAILED!")
        
    return 0 if success else 1

if __name__ == "__main__":
    main()
