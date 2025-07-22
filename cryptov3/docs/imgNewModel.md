# imgNewModel Feature

The imgNewModel is an advanced cryptocurrency image analysis feature that combines traditional chart analysis with survey-based market predictions.

## Components

### Frontend
- **Page Component**: `src/app/imgNewModel/page.tsx`
  - Image upload interface
  - Timeframe selection
  - Optional prediction input
  - Results display with tabbed interface

- **Helper Functions**: `src/lib/imgNewModel-helpers.ts`
  - TypeScript interfaces for predictions
  - Image processing utilities
  - Data formatting functions

- **Survey Processor**: `src/lib/SurveyModelProcessor.ts`
  - Survey data processing and integration
  - Confidence score calculations

### API
- **Route Handler**: `src/app/api/imgNewModel/route.ts`
  - Handles POST requests with form data
  - Processes uploaded images
  - Communicates with Python backend
  - Returns formatted responses

### Backend
- **FastAPI Server**: `backend/app.py`
  - Image analysis endpoints
  - Chart detection logic
  - Survey data integration

- **Support Files**:
  - `backend/examples/chart_analysis.py`: Chart pattern detection
  - `backend/examples/generate_survey_json.py`: Survey data generator
  - `backend/examples/demo_simplified.py`: Testing utilities

## How It Works

1. Users upload cryptocurrency chart images
2. Images are analyzed to determine if they contain charts
3. For chart images:
   - Technical indicators are extracted
   - Pattern recognition is applied
   - Survey data is integrated for enhanced predictions
4. For non-chart images:
   - Survey-based predictions are generated
5. Results include:
   - Market movement prediction (UP/DOWN)
   - Confidence score (percentage)
   - Technical indicators (for chart images)
   - Survey-based insights

## Usage

1. Navigate to `/imgNewModel` in your browser
2. Upload a cryptocurrency chart image
3. Select the appropriate timeframe
4. Add any additional prediction context (optional)
5. View the analysis results in the tabbed interface

## Technical Details

The feature uses a hybrid approach combining:
- Image processing techniques for chart detection
- Pattern recognition algorithms for technical analysis
- Survey data processing for market sentiment
- Combined scoring for final predictions

The confidence scores are calculated based on both technical indicators and survey data, weighted according to the reliability of each source.
