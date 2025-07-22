# Candlestick Chart Prediction Platform

An AI-powered platform that analyzes candlestick chart patterns and provides trading predictions (BUY/SELL/HOLD).

## Features

- Upload candlestick chart images
- Select timeframe for analysis (15m, 1h, 4h, 1d, 7d)
- AI-powered pattern recognition
- Prediction results with confidence scores
- Detection of common chart patterns
- History of past predictions
- **NEW**: Advanced image analysis with survey-based predictions (imgNewModel)
- **IMPROVED**: Consistent chart analysis results for the same image

## Project Structure

- **Frontend**: Next.js 14 with App Router
  - UI Components: shadcn/ui
  - Animations: framer-motion
  - API Client: axios

- **Backend**: FastAPI + TensorFlow/Keras
  - Image processing with OpenCV
  - REST API for predictions

## Getting Started

### Frontend

1. Install dependencies:

```bash
npm install
```

2. Set environment variables:
Create a `.env.local` file with:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:

```bash
npm run dev
```

### Backend

1. Start the backend server using the provided scripts:

**Windows (PowerShell):**
```powershell
.\start_backend_server.ps1
```

**Windows (Command Prompt):**
```cmd
start_backend_server.bat
```

2. For a quick demo of the prediction algorithms, run:

```bash
.\run_demo.bat
```

This will set up a Python environment and run a simplified demo showing:
- Data preprocessing
- Technical indicators calculation
- Candlestick pattern detection
- Time series models

2. For the full application, navigate to the backend directory:

```bash
cd backend
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

- Windows:
```bash
venv\Scripts\activate
```

- Linux/Mac:
```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. Open your browser to `http://localhost:3000`
2. Upload a candlestick chart image
3. Select the appropriate timeframe
4. Click "Analyze Chart"
5. View the prediction results and detected patterns

## Deployment

- Frontend: Deploy to Vercel
- Backend: Deploy to AWS EC2 or PythonAnywhere
