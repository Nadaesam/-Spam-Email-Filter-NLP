!pip install numpy pandas seaborn matplotlib nltk gensim

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import gensim
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
nltk.download('stopwords')
nltk.download('punkt')

# Load file
df= pd.read_csv(r"Spam_Email_Data.csv")
df.head()

def show_df_info(df):

  print("Data inforamation:")
  print(df.info())
  print()

  # Count the occurrences of each value in the 'target' column
  value_counts = df['target'].value_counts()

  # Create a dictionary to map numeric target values to human-readable labels
  target_labels = {0: "number of non spam emails", 1: "number of spam emails"}

  # Apply the label mapping to the index (categories) of the value counts series
  value_counts.index = value_counts.index.map(target_labels)

  # Print the value counts with informative labels for the target categories
  print(value_counts)

  return df

df = show_df_info(df)

def preprocess_data(txt):

    # Remove email addresses
    txt = re.sub(r'\S+@\S+', '', txt)

    # Remove non-alphabetical characters and HTML tags
    txt = re.sub(r'<.*?>', '', txt)

    # Removes special characters, punctuation, and digits
    txt = re.sub(r'[^a-zA-Z\s]', '', txt)

    # Convert to lowercase
    txt = txt.lower()

    # Tokenize
    tokens = word_tokenize(txt)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]


  # Join the stemmed tokens with spaces in between to form a single processed text string
    processed_text= ' '.join(stemmed_tokens)

    return processed_text
    
# Apply Preprocessing
df["text"]=df['text'].apply(preprocess_data)

# show data after preprocessing
df.head()

# shuffle and split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'],df['target'], test_size=0.4, random_state=50)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the training data
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
# Transform the testing data
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Count Vectorization for n-grams
count_vectorizer = CountVectorizer(ngram_range=(2, 2))
x_train_ngram = count_vectorizer.fit_transform(x_train)
x_test_ngram = count_vectorizer.transform(x_test)

# Word2Vec Model
def create_word2vec_model():
  # convert rows to list of strings (sentences)
  sentences= [row.split() for row in x_train]
  # Train a Word2Vec model on the provided sentences
  word2vec_model = Word2Vec(sentences,vector_size=100)
  # Return the Word2Vec model's word vectors
  return word2vec_model.wv

# Create a pre-trained Word2Vec model
word2vec_model = create_word2vec_model()

def word2vec_embedding_row(txt):
  # Split the input text into individual words
  words = txt.split()

  # Compute the mean vector representation of words in the input text
  embedding_txt = np.mean([word2vec_model[w] for w in words if w in word2vec_model], axis=0)

  return embedding_txt

def word2vec_embedding_data(data):
  # Compute Word2Vec embeddings for each text in the input data
  embedding_data = np.array([word2vec_embedding_row(txt) for txt in data])
  return embedding_data

# Text Embedding using Word2Vec
x_train_word2vec = word2vec_embedding_data(x_train)
x_test_word2vec = word2vec_embedding_data(x_test)

# Function to tag documents
def tag_documents(data):
    tagged_documents = []
    for i, text in enumerate(data):
        tagged_documents.append(TaggedDocument(words=text.split(), tags=[i]))
    return tagged_documents

# Doc2Vec Model
def create_and_train_doc2vec_model():
  # Tag documents in the training corpus
  tagged_train_documents = tag_documents(x_train)
  # Initialize Doc2Vec model
  doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)
  # Build vocabulary
  doc2vec_model.build_vocab(tagged_train_documents)
  # Train Doc2Vec model
  doc2vec_model.train(tagged_train_documents, total_examples=doc2vec_model.corpus_count, epochs=10)

  return doc2vec_model

# Create a pre-trained Doc2Vec model
doc2vec_model = create_and_train_doc2vec_model()

# Function to generate document embeddings
def doc2vec_embedding(data):
    embeddings = []
    for text in data:
        embeddings.append(doc2vec_model.infer_vector(text.split()))
    return embeddings

# Generate document embeddings for training and testing data
x_train_doc2vec = doc2vec_embedding(x_train)
x_test_doc2vec = doc2vec_embedding(x_test)

def logistic_regression(_x_train, y_train, _x_test):

  # Create the logistic regression model
  model = LogisticRegression(C=0.4, random_state=50)

  # Train the model
  model.fit(_x_train, y_train)

  # Make predictions on the testing data
  predictions = model.predict(_x_test)

  return predictions

# Training logistic regression model using word2vec features
logisticRegressionModelPredictionsW2V = logistic_regression(x_train_word2vec, y_train, x_test_word2vec)

# Training logistic regression model using doc2vec features
logisticRegressionModelPredictionsD2V = logistic_regression(x_train_doc2vec, y_train, x_test_doc2vec)

# Training logistic regression model using TF-IDF features
logisticRegressionModelPredictionsTfIdf = logistic_regression(x_train_tfidf, y_train, x_test_tfidf)

# Training logistic regression model using N-gram features
logisticRegressionModelPredictionsNgram = logistic_regression(x_train_ngram, y_train, x_test_ngram)

def decision_Tree(_x_train, y_train, _x_test):

  # Create the decision tree model
  model = DecisionTreeClassifier(random_state=49)

  # Train the model
  model.fit(_x_train, y_train)

  # Make predictions on the testing data
  predictions = model.predict(_x_test)

  return predictions

# Training decision tree model using word2vec features
decisionTreeModelPredictionsW2V = decision_Tree(x_train_word2vec, y_train, x_test_word2vec)

# Training decision tree model using doc2vec features
decisionTreeModelPredictionsD2V = decision_Tree(x_train_doc2vec, y_train, x_test_doc2vec)

# Training decision tree model using TF-IDF features
decisionTreeModelPredictionsW2VTfIdf = decision_Tree(x_train_tfidf, y_train, x_test_tfidf)

# Training decision tree model using N-gram features
decisionTreeModelPredictionsW2VNgram = decision_Tree(x_train_ngram, y_train, x_test_ngram)

# Initialize an empty list to store model evaluation results
model_evaluations = []
#function to calculate and store evaluation metrics
def model_result(modelName, y_test, y_pred):
  # Calculate accuracy and precision
  accuracy_acc = accuracy_score(y_test, y_pred)
  precision_acc = precision_score(y_test, y_pred)
  f1_acc = f1_score(y_test, y_pred)
  recall_acc = recall_score(y_test, y_pred)

 # Create a dictionary to store the evaluation metrics
  model_metrics = {
      'Model': modelName,
      'accuracy_score': accuracy_acc,
      'precision_score': precision_acc,
      'f1_score': f1_acc,
      'recall_score': recall_acc
  }
  model_evaluations.append(model_metrics)

model_result("logistic Regression with word2vec",y_test,logisticRegressionModelPredictionsW2V)
model_result("decision Tree with word2vec",y_test,decisionTreeModelPredictionsW2V)


model_result("logistic Regression with doc2vec",y_test,logisticRegressionModelPredictionsD2V)
model_result("decision Tree with doc2vec",y_test,decisionTreeModelPredictionsD2V)

model_result("logistic Regression with TF IDF",y_test,logisticRegressionModelPredictionsTfIdf)
model_result("decision Tree with TF IDF",y_test,decisionTreeModelPredictionsW2VTfIdf)

model_result("logistic Regression with N-grams",y_test,logisticRegressionModelPredictionsNgram)
model_result("decision Tree with N-grams",y_test,decisionTreeModelPredictionsW2VNgram)


# Print the results
if model_evaluations:
  print("Model evaluation results are : \n")
  df = pd.DataFrame(model_evaluations)
  df['Model'] = df['Model'].apply(lambda x: x.center(40))
  print(df.to_string(justify='center'))
else:
  print("No model evaluation results found.")