# 🧪 Testing Framework

## Overview

This directory contains comprehensive tests for the Twitter Sentiment Analysis project.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual modules
│   ├── test_config.py
│   ├── test_preprocessor.py
│   ├── test_models.py
│   └── test_predictor.py
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   ├── test_data_flow.py
│   └── test_api_integration.py
├── fixtures/                # Test data and fixtures
│   ├── sample_tweets.json
│   ├── mock_config.yaml
│   └── test_model_weights.h5
├── conftest.py             # Pytest configuration
└── __init__.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_preprocessor.py

# Run with verbose output
pytest -v
```

## Test Guidelines

1. **Mock External Dependencies**: Don't rely on Twitter API in tests
2. **Use Fixtures**: Reusable test data and setups
3. **Test Edge Cases**: Handle errors and boundary conditions
4. **Parameterized Tests**: Test multiple scenarios efficiently
5. **Integration Tests**: Test component interactions

## Example Test Structure

```python
# tests/unit/test_preprocessor.py
import pytest
from src.data.preprocessor import TextPreprocessor

class TestTextPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()
    
    def test_clean_text_removes_urls(self, preprocessor):
        text = "Check this out! https://example.com"
        result = preprocessor.clean_text(text)
        assert "https://example.com" not in result
    
    @pytest.mark.parametrize("text,expected", [
        ("I love this!", 1),
        ("This is terrible", 0),
        ("Neutral statement", 0)
    ])
    def test_sentiment_analysis(self, preprocessor, text, expected):
        result = preprocessor.analyze_sentiment(text)
        assert result['binary_sentiment'] == expected
```