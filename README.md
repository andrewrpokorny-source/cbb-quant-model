# CBB Quant Model

College basketball spread prediction model using machine learning.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy env template (optional, for Kalshi integration)
cp .env.example .env
```

## Usage

### Generate Predictions
```bash
uv run python predict.py
```

### Run Dashboard
```bash
uv run streamlit run app.py
```
