# Sentiment Analysis API with BERT

A production-ready sentiment analysis service powered by a fine-tuned BERT model, fully containerized with Docker and featuring a FastAPI backend with a Streamlit web interface.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Troubleshooting](#troubleshooting)

## Project Overview

This project implements a complete sentiment analysis pipeline that:

- **Preprocesses data** from the IMDB dataset with text cleaning and normalization
- **Fine-tunes a DistilBERT model** for binary sentiment classification (positive/negative)
- **Serves predictions** via a REST API built with FastAPI
- **Provides a web UI** using Streamlit for easy interaction
- **Performs batch predictions** on CSV files
- **Is fully containerized** using Docker with orchestration via docker-compose

### Key Features

- **Pre-trained Model**: Uses DistilBERT for faster inference while maintaining high accuracy
- **REST API**: FastAPI-based service with health check and prediction endpoints
- **Web Interface**: Streamlit UI for interactive sentiment analysis
- **Batch Processing**: Script to analyze multiple texts from a CSV file
- **Docker Orchestration**: Complete containerization with health checks
- **MLOps Ready**: Experiment tracking with metrics and run summaries

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  User Interface (Streamlit)           │
│                     Port: 8501                         │
└──────────────────────────────────────────────────────┘
                            │
                            │ HTTP
                            ▼
┌──────────────────────────────────────────────────────┐
│               FastAPI Backend Service                 │
│                     Port: 8000                         │
│  ┌────────────────────────────────────────────────┐  │
│  │   Fine-tuned DistilBERT Model                  │  │
│  │   (Loaded on Startup)                          │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── data/
│   ├── raw/                 # Raw data storage (not used in production)
│   ├── processed/           # Processed train.csv and test.csv
│   └── unseen/             # Data for batch predictions
├── model_output/           # Fine-tuned model artifacts
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── results/                # Evaluation metrics and run summaries
│   ├── metrics.json
│   └── run_summary.json
├── scripts/                # Utility scripts
│   ├── preprocess.py       # Data preprocessing script
│   ├── train.py            # Model training script
│   └── batch_predict.py    # Batch prediction script
├── src/                    # Application source code
│   ├── api.py              # FastAPI application
│   └── ui.py               # Streamlit application
├── tests/                  # Test files (optional)
├── Dockerfile.api          # Container definition for API
├── Dockerfile.ui           # Container definition for UI
├── docker-compose.yml      # Docker orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md              # This file
```

## Prerequisites

- **Docker** (version 20.10+)
- **Docker Compose** (version 1.29+)
- **Git** (for version control)
- **Python 3.10+** (if running locally without Docker)

## Setup Instructions

### Option 1: Using Docker Compose (Recommended)

#### Step 1: Prepare the Model

First, you need to generate the fine-tuned model artifacts. This is a one-time setup:

```bash
# 1. Clone or navigate to the project directory
cd Twitter-sentiment-analysis-API-with-BERT

# 2. Create a Python environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the preprocessing script
python scripts/preprocess.py

# 5. Train the model (this may take several minutes)
#    – by default this uses DistilBERT for speed; to use original BERT run:
#        MODEL_NAME=bert-base-uncased python scripts/train.py
#    (you can also set MODEL_NAME in your environment or .env file)
python scripts/train.py

# Verify model artifacts were created
ls model_output/  # Should show: config.json, pytorch_model.bin, tokenizer_config.json, vocab.txt
```

*Note:* training with `bert-base-uncased` roughly doubles memory usage and
takes longer on CPU/GPU compared to DistilBERT. Choose the model according to
your compute resources and latency requirements.

#### Step 2: Create Environment File

Copy `.env.example` to `.env` (optional, defaults are provided):

```bash
cp .env.example .env
```

#### Step 3: Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build -d

# Monitor the startup
docker-compose logs -f

# Check container health
docker ps
# Both containers should show "healthy" status within 3 minutes
```

#### Step 4: Access the Application

- **Web Interface**: Open http://localhost:8501 in your browser
- **API Documentation**: Visit http://localhost:8000/docs
- **API Health Check**: `curl netstat -ano | findstr :8000
netstat -ano | findstr :8501`

### Option 2: Local Development (Without Docker)

If you prefer to run locally:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python scripts/preprocess.py

# Run training
python scripts/train.py

# In one terminal, start the API
python src/api.py

# In another terminal, start the UI
streamlit run src/ui.py
```

## Usage

### Web Interface

1. Navigate to http://localhost:8501
2. Enter text in the text area
3. Click "Analyze Sentiment" button
4. View the predicted sentiment and confidence score

**Example inputs**:
- "This product is amazing! I love it so much." → Positive
- "Terrible experience, would not recommend." → Negative
- "The movie was okay, nothing special." → May classify as either

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response (200 OK)**:
```json
{
  "status": "ok"
}
```

#### Sentiment Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

**Response (200 OK)**:
```json
{
  "sentiment": "positive",
  "confidence": 0.9876
}
```

**Error Response (400 Bad Request)**:
```json
{
  "detail": "Text field cannot be empty"
}
```

#### Interactive API Documentation

Access Swagger UI: http://localhost:8000/docs

### Batch Prediction

Predict sentiment for multiple texts from a CSV file:

```bash
# Prepare input CSV with 'text' column
python scripts/batch_predict.py \
  --input-file data/unseen/texts.csv \
  --output-file results/predictions.csv
```

**Input CSV format** (`data/unseen/texts.csv`):
```csv
text
"I enjoyed this movie"
"Terrible experience"
"It was fine"
```

**Output CSV format** (`results/predictions.csv`):
```csv
text,predicted_sentiment,confidence
"I enjoyed this movie",positive,0.956
"Terrible experience",negative,0.987
"It was fine",positive,0.512
```

## API Documentation

### Request/Response Format

All API communication uses JSON format.

**Request Headers**:
```
Content-Type: application/json
```

**Response Headers**:
```
Content-Type: application/json
```

### Error Handling

The API returns appropriate HTTP status codes:

- **200 OK**: Successful prediction
- **400 Bad Request**: Invalid input (empty text, missing field)
- **422 Unprocessable Entity**: Malformed JSON body
- **500 Internal Server Error**: Model processing error

## Model Information

### Model Details

- **Base Model**: DistilBERT (distilbert-base-uncased) by default. The training script is parameterized via the `MODEL_NAME` environment variable, so you can substitute the original `bert-base-uncased` or any other compatible Transformer checkpoint if preferred.
- **Training Dataset**: IMDB Movie Reviews (50,000 samples)
- **Task**: Binary Sentiment Classification (Positive/Negative)
- **Framework**: Transformers (Hugging Face)

### Model Selection Rationale

**Why DistilBERT instead of BERT?**
- **Faster Inference**: 40% faster than BERT
- **Smaller Size**: 40% smaller model (66M parameters vs 110M)
- **Performance**: Only ~3-5% accuracy loss compared to BERT
- **Better for Production**: Ideal for real-time API serving

### Training Configuration

```json
{
  "hyperparameters": {
    "model_name": "distilbert-base-uncased",
    "learning_rate": 5e-5,
    "batch_size": 16,
    "num_epochs": 3
  }
}
```

### Expected Performance

Based on the IMDB dataset:
- **Accuracy**: ~92-95%
- **Precision**: ~92-94%
- **Recall**: ~92-96%
- **F1-Score**: ~92-95%

These metrics are logged in `results/metrics.json` and `results/run_summary.json` after training.

## Docker Deployment

### Building Manually

```bash
# Build API image
docker build -f Dockerfile.api -t sentiment-api:latest .

# Build UI image
docker build -f Dockerfile.ui -t sentiment-ui:latest .

# Run API
docker run -p 8000:8000 \
  -v $(pwd)/model_output:/app/model_output:ro \
  sentiment-api:latest

# Run UI (in another terminal)
docker run -p 8501:8501 \
  -e API_URL=http://localhost:8000 \
  sentiment-ui:latest
```

### Health Checks

Both containers include health checks that verify:
- API: Can reach `/health` endpoint
- UI: Can reach Streamlit server on port 8501

Health checks run every 30 seconds with a 40-second startup period.

## Environment Variables

All configurable settings via `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8000 | Port for FastAPI service |
| `UI_PORT` | 8501 | Port for Streamlit service |
| `MODEL_PATH` | /app/model_output | Path to model artifacts in container |
| `API_URL` | http://api:8000 | URL for UI to reach API |
| `STREAMLIT_SERVER_HEADLESS` | true | Run Streamlit in headless mode |

## Troubleshooting

### Containers Won't Start

**Issue**: "Error response from daemon"

**Solution**:
```bash
# Ensure Docker daemon is running
docker --version

# Clean up and retry
docker-compose down --volumes
docker-compose up --build
```

### Model Not Found

**Issue**: "Error loading model: [Errno 2] No such file or directory"

**Solution**: 
```bash
# Verify model_output directory exists and is not empty
ls -la model_output/

# If empty, retrain the model
python scripts/train.py
```

### 500 Error During Prediction

**Issue**: When submitting text from the UI or API, you may see a message similar to:
```
Error 500 - {"detail":"Prediction failed: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'"}
```
This happens because the tokenizer returns `token_type_ids` by default, but some models such as DistilBERT ignore them and raise an error when they are passed to `forward()`.

**Solution**: The API now strips out `token_type_ids` before calling the model. If you fork this project or run your own model, ensure your prediction code removes unsupported inputs:
```python
inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
inputs.pop('token_type_ids', None)  # safe remove
outputs = model(**inputs)
```
For custom or newer models, consult the Hugging Face model documentation to see which inputs are accepted.

### Handling Negations and Low Confidence

Some short phrases containing negated sentiments (e.g. “I don\'t like this movie”) are tricky for the pretrained model, which may assign a weak positive score. To improve robustness, the API now applies a lightweight rule-based classifier when the model’s confidence is below 60 % or when its prediction differs from the rule-based opinion.

The rule-based detector also includes a few simple patterns for negated positive words such as “don\'t like”, “do not like”, and “dislike”, so that the example above returns a sensible negative result:
```python
NEGATIVE_PATTERNS = [r"\bdon't like\b", r"\bdo not like\b", r"\bdislike\b"]

# in predict():
if confidence < 0.6:
    rb_sent, rb_conf = rule_based_predict(text)
    if rb_sent != sentiment and rb_conf > confidence:
        sentiment, confidence = rb_sent, rb_conf
```

This heuristic boosts accuracy on simple negations and gives you a fallback if the fine-tuned model is uncertain. You can extend the pattern list or adjust the confidence threshold to suit your data.

### API and UI Can't Communicate

**Issue**: "Could not connect to the API"

**Solution**:
```bash
# Check if API container is healthy
docker ps
# Both containers should show "healthy" in STATUS

# Check API logs
docker logs sentiment-api

# Verify network connectivity
docker exec sentiment-ui ping api
```

### Out of Memory During Training

**Issue**: "CUDA out of memory" or "RuntimeError: CUDA out of memory"

**Solution**: Reduce batch size in `scripts/train.py`:
```python
per_device_train_batch_size=8,  # Reduce from 16
per_device_eval_batch_size=8,   # Reduce from 16
```

### Port Already in Use

**Issue**: "Address already in use"

**Solution**:
```bash
# Change ports in .env file
API_PORT=8001
UI_PORT=8502

# Or stop the service using the port
# On Linux/Mac
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# On Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Streamlit Not Loading

**Issue**: Page keeps loading or times out

**Solution**:
```bash
# Check Streamlit logs
docker logs sentiment-ui

# Ensure API is healthy
curl http://localhost:8000/health

# Restart services
docker-compose restart
```

## Performance Optimization

### API Response Time

Typical inference time on CPU: **0.5 - 1.5 seconds**
Typical inference time on GPU: **0.1 - 0.3 seconds**

### Memory Requirements

- **Model Size**: ~260 MB (DistilBERT)
- **API Container**: ~500 MB RAM
- **UI Container**: ~300 MB RAM
- **Total**: ~1 GB minimum

### Production Recommendations

1. **Use GPU**: Add GPU support in `docker-compose.yml` for faster inference
2. **Model Caching**: Model is loaded once at API startup (efficient)
3. **Load Balancing**: Use Nginx or similar for multiple API instances
4. **Monitoring**: Implement Prometheus metrics collection
5. **Model Versioning**: Tag and version model artifacts

## File Descriptions

### Core Scripts

- **scripts/preprocess.py**: Downloads IMDB dataset, cleans text, splits into train/test
- **scripts/train.py**: Fine-tunes DistilBERT, saves artifacts, logs metrics
- **scripts/batch_predict.py**: Bulk prediction on CSV files

### Application Code

- **src/api.py**: FastAPI service with `/health` and `/predict` endpoints
- **src/ui.py**: Streamlit interface for interactive predictions

### Configuration

- **requirements.txt**: Python package dependencies (torch, transformers, fastapi, streamlit)
- **Dockerfile.api**: Multi-stage Docker build for API service
- **Dockerfile.ui**: Multi-stage Docker build for UI service
- **docker-compose.yml**: Orchestrates API and UI services with health checks
- **.env.example**: Template for environment variables

## Git Configuration

Recommended `.gitignore` entries:

```
# Virtual environment
venv/
env/

# Model artifacts (large files)
model_output/
*.bin
*.pt

# Training outputs
training_output/
logs/

# Data
data/raw/
data/processed/
data/unseen/

# Results
results/

# Python cache
__pycache__/
*.pyc
*.pyo

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Environment
.env
```

## Support and Contributing

For issues or improvements:

1. Check the Troubleshooting section
2. Review container logs with `docker logs`
3. Verify model artifacts exist in `model_output/`
4. Ensure all dependencies are installed

## License

This project is provided as-is for educational and commercial use.

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [IMDB Dataset](https://huggingface.co/datasets/imdb)
