import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_csv(csv_file, exp):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    if exp == 1: # just in case we need to hurry with experiments and just want a fraction of dataset cls
        
        subset_df = df.sample(frac=1, random_state=42)
        df = subset_df
    
    # Iterate over the email column and preprocess each email
    processed_texts = []
    for email in df['email']:
        # Convert to lowercase
        email = email.lower()
        
        # Remove special characters and numbers
        email = re.sub(r'[^a-zA-Z]', ' ', email)
        
        # Tokenize the text into words
        tokens = word_tokenize(email)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Apply stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        # Join the processed tokens back into a single string
        processed_text = ' '.join(tokens)
        
        # Add the processed text to the list
        processed_texts.append(processed_text)
    
    # Add a new column to the DataFrame with the preprocessed texts and delete the old one
    df['processed_text'] = processed_texts
    df = df.drop('email', axis=1)
    
    # Return the updated DataFrame
    return df
