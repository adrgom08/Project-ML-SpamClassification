import numpy as np
from Preprocessing import preprocess_csv

# Using unweightened K Nearest Neighbours for classification of emails
from sklearn.neighbors import KNeighborsClassifier
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

# Train a Knn classifier
knn_classifier =KNeighborsClassifier() #Default value, 5 neighbours
knn_classifier.fit(X_train_transformed, y_train)

# Make predictions on the testing set
y_pred = knn_classifier.predict(X_test_transformed)

# Evaluate the accuracy of the classifier (it is the important evaluation parameter)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Knn(5 neighbours, uniform): ", accuracy)