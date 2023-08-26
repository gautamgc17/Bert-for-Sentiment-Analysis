## Twitter US Airline Sentiment Analysis 

This project demonstrates sentiment analysis on tweets related to US airlines using the BERT model into 3 classes - positive, negative and neutral. The dataset used for this project is the [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment), which contains tweets labeled as positive, neutral, or negative sentiments for six major US airlines.


## Overview

In this project, we finetune the pre-trained BERT (Bidirectional Encoder Representations from Transformers) model to perform sentiment analysis on tweets. The main steps of the project are as follows:

1. **Data Preprocessing**: Tweets are cleaned, removing links, usernames, emojis, and non-alphabetic characters.

2. **Label Encoding**: Sentiment labels ("positive", "neutral", "negative") are encoded into numerical values.

3. **Model**: A pre-trained BERT model for sequence classification is loaded and fine-tuned for the sentiment analysis task.

4. **Training**: The model is trained using a custom dataset containing tweet text and encoded labels.

5. **Evaluation**: Model performance is evaluated on a test set using different classification metrics such as accuracy, f1-score.


## Prerequisites

Before running the python application, we'll need the following:

- Python 3.10.12 installed
- Necessary packages (install using pip install -r requirements.txt)


## Getting Started

**Step 1. Clone the repository to your local machine and then switch to code directory**

```
git clone https://github.com/gautamgc17/Bert-for-Sentiment-Analysis.git
cd Bert-for-Sentiment-Analysis
```

**Step 2. Create a Virtual Environment and install Dependencies.**

```
pip install virtualenv
```

Create a new Virtual Environment for the project and activate it.

```
virtualenv env
env\Scripts\activate
```
Now install the project dependencies in this virtual environment, which are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

**Step3. Download dataset**

Download the [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) and save it as `Tweets.csv` in the repository's root directory.


**Step 4. Run the Python application for model training**

```
python run sentiment_analysis.py
```

## Results

![confusion_matrix]()

