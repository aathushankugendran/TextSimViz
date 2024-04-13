# Importing necessary libraries
import nltk  # Natural Language Toolkit
import pandas as pd
from nltk.tokenize import word_tokenize  # Tokenization function
from nltk.stem import PorterStemmer  # Porter Stemmer for stemming
from nltk.stem import WordNetLemmatizer  # WordNet Lemmatizer for lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # TF-IDF and Count Vectorizer
from sklearn.cluster import KMeans  # KMeans clustering algorithm
from wordcloud import WordCloud  # WordCloud visualization
import matplotlib.pyplot as plt  # Matplotlib for plotting
from kneed import KneeLocator  # KneeLocator for finding the elbow point
from gensim.models import Word2Vec  # Word2Vec model for word embeddings

# Downloading NLTK resources
nltk.download('punkt')  # Download NLTK tokenizer data
nltk.download('wordnet')  # Download NLTK WordNet lemmatizer data

# Given sentences
df = pd.read_csv('nlp-dataset.csv')
sentences = df['text'].tolist()

# Step 1: Tokenization using NLTK word_tokenize
tokenized_sentences = []

for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence.lower())  # Tokenize the sentence and convert to lowercase
    tokenized_sentences.append(tokenized_sentence)  # Append the tokenized sentence to the list

# Step 2: Stemming using NLTK PorterStemmer
porter_stemmer = PorterStemmer()  # Initialize the PorterStemmer
stemmed_sentences = []

for sentence in tokenized_sentences:
    stemmed_sentence = []  # Initialize an empty list to store stemmed words
    for word in sentence:
        stemmed_word = porter_stemmer.stem(word)  # Stem each word in the sentence
        stemmed_sentence.append(stemmed_word)  # Append the stemmed word to the list
    stemmed_sentences.append(stemmed_sentence)  # Append the stemmed sentence to the list

# Step 3: Lemmatization using NLTK WordNetLemmatizer
lemmatizer = WordNetLemmatizer()  # Initialize the WordNetLemmatizer
lemmatized_sentences = []

for sentence in stemmed_sentences:
    lemmatized_sentence = []  # Initialize an empty list to store lemmatized words
    for word in sentence:
        lemmatized_word = lemmatizer.lemmatize(word)  # Lemmatize each word in the sentence
        lemmatized_sentence.append(lemmatized_word)  # Append the lemmatized word to the list
    lemmatized_sentences.append(lemmatized_sentence)  # Append the lemmatized sentence to the list
    
# Step 4: Feature extraction
# a. TF-IDF
tfidf_vectorizer = TfidfVectorizer()  # Initialize the TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(sentence) for sentence in lemmatized_sentences])  # Fit and transform the lemmatized sentences using TF-IDF

# b. TF (Term Frequency)
tf_vectorizer = CountVectorizer()  # Initialize the Term Frequency vectorizer
tf_matrix = tf_vectorizer.fit_transform([' '.join(sentence) for sentence in lemmatized_sentences])  # Fit and transform the lemmatized sentences using TF

# c. BOW (Bag of Words)
bow_vectorizer = CountVectorizer(binary=True)  # Initialize the Bag of Words vectorizer
bow_matrix = bow_vectorizer.fit_transform([' '.join(sentence) for sentence in lemmatized_sentences])  # Fit and transform the lemmatized sentences using Bag of Words

# d. Word2Vec
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)  # Train Word2Vec model on sentences

# Step 5: Clustering using K-means
if len(sentences) >= 2:
    knee = KneeLocator(
        range(1, min(len(sentences), 10)),
        [KMeans(n_clusters=i, random_state=42).fit(tfidf_matrix).inertia_ for i in range(1, min(len(sentences), 10))],
        curve='convex', direction='decreasing'
    )
    k_value = knee.elbow
else:
    k_value = 1

# Ensure k_value is not None
if k_value is None:
    k_value = 1

# Fitting KMeans with the optimal K value
kmeans = KMeans(n_clusters=k_value, random_state=42)
kmeans.fit(tfidf_matrix)
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Step 6: Visualizing clusters using WordCloud
for cluster in range(k_value):
    cluster_sentences = [sentences[i] for i in range(len(sentences)) if cluster_labels[i] == cluster]  # Select sentences belonging to the current cluster
    cluster_words = ' '.join(cluster_sentences)  # Concatenate sentences to form a single string
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(cluster_words)  # Generate WordCloud for the cluster

    plt.figure(figsize=(8, 4))  # Set the figure size
    plt.imshow(wordcloud, interpolation='bilinear')  # Display
    plt.axis('off')  # Turn off axis
    plt.show()  # Show the word cloud