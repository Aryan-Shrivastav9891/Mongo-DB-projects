"""
Demo script to verify consistent image analysis results

This script loads a sample image and processes it multiple times
to verify that the results are consistent.
"""
import os
import sys
import numpy as np
from PIL import Image
import json
import hashlib

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the chart analysis function
from examples.chart_analysis import analyze_candlestick_chart

def main():
    """Main function to test consistency of analysis"""
    print("Testing consistency of chart analysis...")
    
    # Check if a sample image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default test image if available
        default_paths = [
            "test_images/chart_sample.png",
            "test_images/chart_sample.jpg",
            "../test_images/chart_sample.png",
            "../test_images/chart_sample.jpg",
        ]
        
        image_path = None
        for path in default_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            print("No test image found. Please provide an image path as an argument.")
            print("Usage: python test_consistency.py path/to/image.png")
            return
    
    # Load the image
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        print(f"Loaded image: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Test with different timeframes
    timeframes = ['15m', '1h', '4h', '1d', '7d']
    
    print("\nTesting consistency across timeframes:")
    print("--------------------------------------")
    
    # First run for each timeframe
    first_run_results = {}
    for timeframe in timeframes:
        result = analyze_candlestick_chart(img_array, timeframe)
        first_run_results[timeframe] = result
        print(f"\nTimeframe: {timeframe}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Patterns: {', '.join(result['detected_patterns'])}")
        print(f"Image Hash: {result['image_hash']}")
    
    # Test consistency by running multiple times
    print("\n\nTesting consistency with multiple runs:")
    print("--------------------------------------")
    
    timeframe = '1h'  # Use 1h for consistency test
    print(f"\nTimeframe: {timeframe}")
    print("Running 5 times to verify consistency...")
    
    # Store results for comparison
    all_results = []
    
    # Run 5 times
    for i in range(5):
        result = analyze_candlestick_chart(img_array, timeframe)
        all_results.append(result)
        print(f"\nRun {i+1}:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Patterns: {', '.join(result['detected_patterns'])}")
    
    # Check if all results are the same
    consistent = True
    first_result = all_results[0]
    for i, result in enumerate(all_results[1:], 2):
        if (result['prediction'] != first_result['prediction'] or 
            result['confidence'] != first_result['confidence'] or 
            result['detected_patterns'] != first_result['detected_patterns']):
            consistent = False
            print(f"\nInconsistency detected in run {i}!")
    
    if consistent:
        print("\nSUCCESS: All runs produced consistent results!")
    else:
        print("\nFAILURE: Inconsistent results detected!")

if __name__ == "__main__":
    main()
