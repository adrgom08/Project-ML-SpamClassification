import numpy as np
from Preprocessing import preprocess_csv

# Using Ensemble (voting) with SVC, Naive Bayes and Knnclassification of emails
# the algorithm is imported from the sklearn library
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
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

# Initialize classifiers
knn_classifier = KNeighborsClassifier(n_neighbors=2)
knn_classifier.fit(X_train_transformed, y_train)
svm_classifier = SVC(kernel='linear', probability=True, C=3)
svm_classifier.fit(X_train_transformed, y_train)
nb_classifier = MultinomialNB(alpha = 0.01)
nb_classifier.fit(X_train_transformed, y_train)

# Create ensemble using VotingClassifier
ensemble_classifier = VotingClassifier(estimators=[
    ('knn', knn_classifier),
    ('svm', svm_classifier),
    ('nb', nb_classifier)
], voting='soft')

ensemble_classifier.fit(X_train_transformed, y_train)
# Make predictions on the testing set
y_pred = nb_classifier.predict(X_test_transformed)

# Evaluate the accuracy of the classifier (it is the important evaluation parameter)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Ensemble (SVC, KNN, Naive Bayes): ", accuracy)