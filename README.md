# Twitter Sentiment Analysis for Indian Elections

A comprehensive machine learning project that analyzes Twitter sentiment data to predict election outcomes for Indian elections, focusing on the two major political parties: BJP and Congress.

## 🔍 Project Overview

This project implements an end-to-end machine learning pipeline for sentiment analysis of tweets related to Indian elections. It includes data collection via Twitter API, preprocessing, sentiment labeling, exploratory data analysis, and deployment of deep learning models for sentiment prediction.

## 📁 Project Structure

The repository contains comprehensive Jupyter notebooks covering the complete workflow:

### 📖 Notebooks

1. **`Portfolio.ipynb`** - Complete project overview and demonstration
2. **`Election_dataScrape.ipynb`** - Twitter data collection using Tweepy API
3. **`Data_Labeling .ipynb`** - Data cleaning, sentiment labeling, and dataset balancing
4. **`Exploratory Data Analysis.ipynb`** - Data visualization and pattern analysis
5. **`Election_Glove.ipynb`** - GloVe word embeddings model implementation
6. **`Bidirectional_LSTM.ipynb`** - Advanced bidirectional LSTM model (best performer)
7. **`Tweet_Predictions.ipynb`** - Final predictions using trained models

### 📊 Data & Models

* **`Data.7z`** - Raw and processed datasets (including `Election.csv`)
* **`SavedModels.7z`** - Trained model weights and architectures

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn
pip install tensorflow keras
pip install tweepy textblob nltk
pip install matplotlib seaborn wordcloud
pip install jupyter notebook
```

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/fusebox440/indian_election.git
   cd indian_election
   ```

2. **Twitter API Setup**
   - Create a Twitter Developer account
   - Generate API keys and access tokens
   - Update credentials in the notebook files:
     ```python
     consumer_key = 'DEPMCrTEZmnRPTD0brKmK8aaD'
     consumer_secret = 'SXGlGzeIp0xURNFBa2vzclKuXQPIUKSQuzRzoxlmGHEwF0cx12'
     access_token = '1565933303756378113-6TWZVSVbe6JpziuFoT4r7Y4r5s9nCJ'
     access_token_secret = '4pev1yRXdPcKRheGxRb0Y5T9k74hAmFuWl8WLo3xMpUCn'
     ```

3. **Download GloVe Embeddings**
   - Download from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
   - Extract `glove.6B.100d.txt` to `./glove/` directory

4. **Extract Data Files**
   ```bash
   # Extract the compressed data files
   7z x Data.7z
   7z x SavedModels.7z
   ```

## 🔧 Technical Architecture

### Data Pipeline
1. **Collection**: Twitter API → Raw tweets
2. **Cleaning**: Remove URLs, RT tags, emoticons, stopwords
3. **Labeling**: TextBlob sentiment analysis → Binary classification
4. **Balancing**: Address class imbalance through resampling
5. **Tokenization**: Convert text to numerical sequences
6. **Modeling**: Train deep learning models
7. **Prediction**: Generate sentiment forecasts

### Models Implemented
- **GloVe + Dense Neural Network**: Baseline model with pre-trained embeddings
- **Bidirectional LSTM**: Best performing model using sequence processing

### Key Technologies
- **Data Collection**: Tweepy (Twitter API)
- **NLP Processing**: NLTK, TextBlob, Keras preprocessing
- **Machine Learning**: TensorFlow/Keras, scikit-learn
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Data Handling**: Pandas, NumPy

## 📈 Results

### Model Performance
- **Bidirectional LSTM**: Best performing model with 97%+ accuracy
- **GloVe Model**: Strong baseline performance
- **Prediction Results**: BJP (660 positive tweets) vs Congress (416 positive tweets)

### Key Insights
- Majority of tweets showed negative sentiment for both parties
- Geographic patterns identified key constituencies
- Word clouds revealed important political figures and issues

## 🛠️ Usage

### Running the Analysis

1. **Data Collection**
   ```bash
   jupyter notebook Election_dataScrape.ipynb
   ```

2. **Data Preprocessing**
   ```bash
   jupyter notebook "Data_Labeling .ipynb"
   ```

3. **Exploratory Analysis**
   ```bash
   jupyter notebook "Exploratory Data Analysis.ipynb"
   ```

4. **Model Training**
   ```bash
   # For GloVe model
   jupyter notebook Election_Glove.ipynb
   
   # For Bidirectional LSTM (recommended)
   jupyter notebook Bidirectional_LSTM.ipynb
   ```

5. **Making Predictions**
   ```bash
   jupyter notebook Tweet_Predictions.ipynb
   ```

## ⚠️ Important Notes

### Limitations
- **Data Bias**: Twitter represents primarily urban, educated demographics
- **Sample Size**: Limited dataset size for comprehensive election prediction
- **Temporal Scope**: Analysis limited to specific time periods in 2019
- **Platform Limitation**: Twitter sentiment may not reflect overall population

### Security Considerations
- Never commit API credentials to version control
- Use environment variables for sensitive data
- Follow Twitter's API terms of service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Lakshya Khetan**
- GitHub: [@fusebox440](https://github.com/fusebox440)
- Email: lakshyaketan00@gmail.com

## 📝 Future Scope

- **Enhanced Data Collection**: Incorporate Facebook, news websites
- **Advanced Models**: Implement BERT, attention mechanisms
- **Real-time Analysis**: Live sentiment monitoring
- **Geographic Analysis**: State-wise sentiment mapping
- **Multi-language Support**: Regional language processing

## 📚 References

* [Tweepy Documentation](http://docs.tweepy.org/)
* [Twitter Sentiment Analysis Guide](https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90)
* [Text Data Cleaning in Python](https://www.analyticsvidhya.com/blog/2015/06/quick-guide-text-data-cleaning-python/)
* [Handling Imbalanced Classes](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
* [WordCloud in Python](https://www.datacamp.com/community/tutorials/wordcloud-python)
* [Stanford GloVe](https://github.com/stanfordnlp/GloVe)

## 📄 License

[MIT License](LICENSE.MD)

---

**⭐ If you found this project helpful, please consider giving it a star!**
