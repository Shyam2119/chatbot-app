# models/intent_model.py
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

class IntentClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('data/intents.json').read())
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']
        self.model = None
        
    def preprocess_data(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        self.words = [self.lemmatizer.lemmatize(word) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        
        # Save words and classes for later use
        pickle.dump(self.words, open('models/words.pkl', 'wb'))
        pickle.dump(self.classes, open('models/classes.pkl', 'wb'))
        
        # Create training data
        training = []
        output_empty = [0] * len(self.classes)
        
        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            for word in self.words:
                bag.append(1 if word in word_patterns else 0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])
        
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = np.array([item[0] for item in training])
        train_y = np.array([item[1] for item in training])
        
        return train_x, train_y
    
    def build_model(self):
        train_x, train_y = self.preprocess_data()
        
        # Build neural network model
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))
        
        # Compile model
        self.model.compile(loss='categorical_crossentropy', 
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                      metrics=['accuracy'])
        
        # Train and save model
        self.model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
        self.model.save('models/intent_model.h5')
        print("Model trained and saved")
    
    def predict_intent(self, message):
        # Load model if not already loaded
        if not self.model:
            self.model = tf.keras.models.load_model('models/intent_model.h5')
            self.words = pickle.load(open('models/words.pkl', 'rb'))
            self.classes = pickle.load(open('models/classes.pkl', 'rb'))
        
        # Preprocess input
        sentence_words = nltk.word_tokenize(message)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        
        # Create bag of words
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        
        # Get prediction
        result = self.model.predict(np.array([bag]))[0]
        
        # Get highest probability
        results = [[i, r] for i, r in enumerate(result)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return intent with highest probability if above threshold (0.6)
        if results[0][1] > 0.6:
            return self.classes[results[0][0]]
        else:
            return "unknown"