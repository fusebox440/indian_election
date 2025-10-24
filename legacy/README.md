# Legacy Code Archive

This directory contains the original notebook files from the initial project structure. These files have been moved here to maintain project history while keeping the main repository clean and organized.

## Original Notebooks

- **Portfolio.ipynb** - Original project overview and demonstration notebook
- **Bidirectional_LSTM.ipynb** - LSTM model implementation
- **Data_Labeling.ipynb** - Data labeling and preprocessing workflows
- **Election_dataScrape.ipynb** - Twitter data collection scripts
- **Election_Glove.ipynb** - GloVe embeddings implementation
- **Exploratory Data Analysis.ipynb** - Data exploration and visualization
- **Tweet_Predictions.ipynb** - Model predictions and evaluation
- **download_glove.py** - Utility script for downloading GloVe embeddings

## Migration to New Structure

These notebooks have been refactored and their functionality has been extracted into the modular `src/` package structure:

- **Data Collection**: `src/data/collector.py`
- **Data Preprocessing**: `src/data/preprocessor.py`
- **Model Architecture**: `src/models/sentiment_models.py`
- **Predictions**: `src/models/predictor.py`
- **Utilities**: `src/utils/`

## Clean Notebooks

The new structured approach provides clean notebooks in the `notebooks/` directory that demonstrate the workflow using the modular components.

---
*These legacy files are preserved for reference and historical purposes.*