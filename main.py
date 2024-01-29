from Preprocessing import preprocess_csv

# Using Ensemble (voting) with SVC, Naive Bayes and Knnclassification of emails
# the algorithm is imported from the sklearn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

def train_test(train_csv,test_csv, predictions_txt):
    # Read the training and testing CSV files
    train_data = preprocess_csv(train_csv, 0)
    test_data = preprocess_csv(test_csv, 0)

    # Extract the preprocessed text and labels from the training data
    train_email = train_data['processed_text']
    y_train = train_data['label']

    # Extract the preprocessed text from the testing data
    test_text = test_data['processed_text']

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(train_email)

    # Transform the testing text using the trained vectorizer
    X_test_vectorized = vectorizer.transform(test_text)


    # Due to the high dimensionality, we feature the selection for it to be less expensive
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X_train_vectorized, y_train)
    X_train_transformed = selector.transform(X_train_vectorized).toarray()
    X_test_transformed  = selector.transform(X_test_vectorized).toarray()

    # Initialize classifiers
    rf_classifier = RandomForestClassifier(n_estimators=100)
    rf_classifier.fit(X_train_transformed, y_train)
    svm_classifier = SVC(kernel='linear', probability=True, C=3)
    svm_classifier.fit(X_train_transformed, y_train)
    mlp_classifier = MLPClassifier(learning_rate='adaptive', activation='logistic') 
    mlp_classifier.fit(X_train_transformed, y_train)

    # Create ensemble using VotingClassifier
    ensemble_classifier = VotingClassifier(estimators=[
        ('rf', rf_classifier),
        ('nn', mlp_classifier),
        ('svm', svm_classifier)
    ], voting='soft')

    ensemble_classifier.fit(X_train_transformed, y_train)
    # Make predictions on the testing set
    y_pred = ensemble_classifier.predict(X_test_transformed)

    # Write the predictions to a TXT file
    with open(predictions_txt, 'w') as file:
        for prediction in y_pred:
            file.write(str(prediction) + '\n')

    print(f"Predictions saved to {predictions_txt}.")


# Call the preprocess_csv function, default is the train.csv
train_csv = "train.csv"
test_csv = "train.csv"
predictions_txt = "predictions.txt"

train_test(train_csv, test_csv, predictions_txt)


 

    
