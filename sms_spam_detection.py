import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def load_dataset():
    """Load the SMS Spam Collection Dataset."""
    dataset_path = "customCollection.csv"  # Local path to the dataset
    dataset = pd.read_csv(dataset_path, sep='|', header=None, names=['Label', 'Message'],engine='python')
    dataset['Label'] = dataset['Label'].map({'ham': 'Not Spam', 'spam': 'Spam'})
    dataset['Message'] = dataset['Message'].fillna('') 
    print(dataset.head())
    return dataset

def train_model():
    """Train the spam classifier model and visualize results."""
    dataset = load_dataset()
    X = dataset['Message']
    y = dataset['Label']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Performance:")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

    # Visualizations
    conf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=['Not Spam', 'Spam'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Save the model and vectorizer
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def predict_message():
    """Classify a user-provided message as spam or not."""
    if not (os.path.exists('spam_classifier_model.pkl') and os.path.exists('vectorizer.pkl')):
        print("Model or Vectorizer not found. Training a new model...")
        train_model()

    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    while True:
        user_input = input("Enter a message to classify (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        print("\n================ Result ================")
        print(f"Message: {user_input}")
        print(f"Classification: {prediction}")
        print("=======================================\n")
        
def predict_pile_messages():
    """Classify a batch of messages from a file as spam or not."""
    if not (os.path.exists('spam_classifier_model.pkl') and os.path.exists('vectorizer.pkl')):
        print("Model or Vectorizer not found. Training a new model...")
        train_model()

    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    file_path = input("\nEnter the path to the file containing messages (e.g., .csv): ")
    if not os.path.exists(file_path):
        print("File not found. Please check the path and try again.")
        return

    try:
        # Read the file
        messages = pd.read_csv(file_path, header=None, names=['Message'],sep=',',engine='python')
        print(f"\nLoaded {len(messages)} messages from the file.")
        # print(messages.head())
        
        # Check for empty messages
        messages['Message'] = messages['Message'].astype(str).str.strip()
        messages = messages[messages['Message'] != '']
        # print(messages.head())

        if messages.empty:
            print("The file contains no valid messages. Please check the content.")
            return

        # Transform and predict
        transformed_messages = vectorizer.transform(messages['Message'])
        predictions = model.predict(transformed_messages)

        # Add predictions to the DataFrame
        messages['Classification'] = predictions

        # Save results to a file
        output_path = "classified_messages.csv"
        messages.to_csv(output_path, index=False)
        print(f"\nClassification completed! Results saved to '{output_path}'.")
    except Exception as e:
        print(f"\nAn error occurred while processing the file: {e}")

    
if __name__ == "__main__":
    print("Welcome to SMS Spam Detection Application")
    print("-------------------------------------------")
    print("Options:")
    print("1. Train Model")
    print("2. Predict Single Message")
    print("3. Predict Pile of Messages")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1/2/3/4): ")

        if choice == '1':
            train_model()
        elif choice == '2':
            predict_message()
        elif choice == '3':
            predict_pile_messages()
        elif choice == '4':
            print("Exiting application!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
