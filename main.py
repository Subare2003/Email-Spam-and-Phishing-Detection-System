import pandas as pd  # For data handling
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.feature_extraction.text import CountVectorizer  # To convert text to numerical format
from sklearn.naive_bayes import MultinomialNB  # The classifier used for spam detection
import tkinter as tk  # GUI library
from tkinter import messagebox  # For showing pop-up messages

# Load dataset from CSV file
spam_df = pd.read_csv("spam.csv")

# Create 'spam' column where 'spam' -> 1 and 'ham' -> 0
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the dataset into training and test sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25)

# Initialize CountVectorizer to convert text messages to numeric form
cv = CountVectorizer()

# Fit the vectorizer on training data and transform it to matrix 
x_train_count = cv.fit_transform(x_train.values)

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(x_train_count, y_train) #fit: learns the relationship between the words in a message and the likelihood that the message is spam or not.

# Function to predict whether a given message is spam or ham
def predict_spam(message):
    message_count = cv.transform([message])  # Convert input message to vector
    prediction = model.predict(message_count)[0]  # Predict using trained model (return a list so take only the first index)
    return "Spam" if prediction == 1 else "Ham"  # Return readable result

# Function to show the accuracy of the model
def show_accuracy():
    x_test_count = cv.transform(x_test)  # Transform test data to match training format
    accuracy = model.score(x_test_count, y_test)  # Get model accuracy
    messagebox.showinfo("Model Accuracy", f"Accuracy: {accuracy:.2%}")  # Show result in a popup

# Function that runs when the "Predict" button is clicked
def on_predict():
    msg = entry.get("1.0", tk.END).strip()  # Get message from text box and strip whitespace
    if msg:
        result = predict_spam(msg)  # Predict result
        result_label.config(text=f"Prediction: {result}")  # Display prediction
    else:
        messagebox.showwarning("Input Error", "Please enter a message.")  # Warn if input is empty

# Initialize main window
root = tk.Tk()
root.title("Spam Detector")  # Set window title

# Add label prompting for input
tk.Label(root, text="Enter your message:").pack(pady=5)

# Text box for entering message
entry = tk.Text(root, height=4, width=50)
entry.pack()

# Button to predict message
tk.Button(root, text="Predict", command=on_predict).pack(pady=5)

# Label to show prediction result
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12, "bold"))
result_label.pack(pady=5)

# Button to show model accuracy
tk.Button(root, text="Show Model Accuracy", command=show_accuracy).pack(pady=5)

# Button to close the application
tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
