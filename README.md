# TextSimViz

## Overview

TextSimViz is a Python project that focuses on analyzing text data to uncover similarities between words and visualize them using WordClouds. It leverages natural language processing (NLP) techniques and clustering algorithms to explore textual data and provide intuitive visualizations of word similarities within the text.

## Features

- **Tokenization:** The text data is tokenized using NLTK's word_tokenize function.
- **Stemming:** Stemming is performed using NLTK's Porter Stemmer to reduce words to their base form.
- **Lemmatization:** NLTK's WordNet Lemmatizer is used to lemmatize words, reducing them to their base dictionary form.
- **Feature Extraction:** Various techniques such as TF-IDF, Term Frequency (TF), and Bag of Words (BOW) are used for feature extraction.
- **Clustering:** K-means clustering algorithm is applied to group similar text data into clusters.
- **Visualization:** WordClouds are generated to visualize the clusters and uncover word similarities within each cluster.

## Installation

To run the project, you need to have Python installed on your system along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install pandas nltk scikit-learn wordcloud matplotlib kneed gensim
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/TextSimViz.git
```

2. Navigate to the project directory:

```bash
cd TextSimViz
```

3. Ensure you have a CSV file named `nlp-dataset.csv` containing the text data.

4. Run the provided Python script:

```bash
python text_sim_viz.py
```

5. Explore the generated WordCloud visualizations to understand word similarities within each cluster.
