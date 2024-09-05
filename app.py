from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model('next_model.keras')  
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# Manually set the max sequence length used during training
max_sequence_length = 18  

# Prediction function
def make_prediction(conv, n_words=1, top_n=3):
    try:
        conv = conv.lower()
        text = conv
        generated_text = text
        top_words_probs = []

        for _ in range(n_words):
            text_tokenize = tokenizer.texts_to_sequences([text])
            text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=max_sequence_length)
            prediction_probs = model.predict(text_padded, verbose=0)[0]
            top_indices = np.argsort(prediction_probs)[-top_n:]
            top_probs = np.sort(prediction_probs)[-top_n:]
            top_words = [list(tokenizer.word_index.keys())[index - 1] for index in top_indices]
            top_words_probs = [(word, float(prob)) for word, prob in zip(top_words, top_probs)]  # Convert float32 to float
            predicted_word_index = np.argmax(prediction_probs)
            predicted_word = list(tokenizer.word_index.keys())[predicted_word_index - 1]
            generated_text += " " + predicted_word
            text += " " + predicted_word

        return top_words_probs
    except Exception as e:
        print(f"Error in make_prediction: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        if not model or not tokenizer:
            raise ValueError("Model or tokenizer not loaded correctly")
        top_words_probs = make_prediction(text)
        return jsonify(top_words_probs)
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
