# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Load the SMS Spam Collection dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
df = pd.read_csv(url, sep='\t', header=None, names=['Label', 'Message'])

# Step 3: Preprocess the data
# Convert labels to binary (ham=0, spam=1)
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Label'], test_size=0.2, random_state=42)

# Convert text messages into a bag of words using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Step 5: Create the prediction function
def predict_message(message):
    message_vect = vectorizer.transform([message])
    prob = model.predict_proba(message_vect)[0][1]
    label = 'spam' if prob >= 0.5 else 'ham'
    return [prob, label]

# Step 6: Test the model with an example message
message = "Congratulations! You've won a free iPhone. Click here to claim your prize."
print(predict_message(message))  # Example test message
