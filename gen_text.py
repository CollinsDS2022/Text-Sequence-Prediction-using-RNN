import tensorflow as tf
import numpy as np
from tensorflow import keras

new_model = tf.keras.models.load_model("hamlet_model.h5")
# Check its architecture
new_model.summary()

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
)


def preprocess(texts):
    X = tokenizer.texts_to_sequences(texts)
    X = np.array(X) - 1
    return tf.one_hot(X, len(tokenizer.word_index))


#     X = np.array(tokenizer.texts_to_sequences(texts)) - 1
#     return tf.one_hot(X, max_id)


def next_char(text, temperature=1):
    X_new = preprocess(text)
    y_proba = new_model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


output = []
for i in np.arange(0, 2.1, 0.1):
    output.append(complete_text("Hamlet", 1000, temperature=i))

with open("your_file.txt", "w+") as f:
    for line in output:
        f.write(f"{line}\n")
