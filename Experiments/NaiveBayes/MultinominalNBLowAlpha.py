import numpy as np
from Preprocessing import preprocess_csv

# Using the Multinomial Bayes algorithm with alpha = 0.01 for classification of emails
# the algorithm is imported from the sklearn library
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

# Preprocessing the emails
csv_file = 'train.csv'
df = preprocess_csv(csv_file, 1)

# initializaing the test and train features and labels
y = df['label']
X = df['processed_text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Vectorize the text data using TF-IDF to convert preprocessed text data 
# (X_train and X_test) into numerical features
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Due to the high dimensionality, we feature the selection for it to be less expensive
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X_train_vectorized, y_train)
X_train_transformed = selector.transform(X_train_vectorized).toarray()
X_test_transformed  = selector.transform(X_test_vectorized).toarray()

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB(alpha = 0.01) #set a very low alpha 
nb_classifier.fit(X_train_transformed, y_train)

# Make predictions on the testing set
y_pred = nb_classifier.predict(X_test_transformed)

# Evaluate the accuracy of the classifier (it is the important evaluation parameter)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Multinomial Naive Bayes(0.01): ", accuracy)