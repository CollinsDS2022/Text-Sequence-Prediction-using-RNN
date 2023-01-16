import tensorflow as tf
import numpy as np
from tensorflow import keras

# hamlet_1_text = open("texts/Fyodor_1.txt", "r", encoding="utf-8").read()
# hamlet_2_text = open("texts/Fyodor_2.txt", "r", encoding="utf-8").read()
# hamlet_3_text = open("texts/Fyodor_3.txt", "r", encoding="utf-8").read()

hamlet_1_text = open('texts/hamlet_1.txt', 'r', encoding="utf-8").read()
hamlet_2_text = open('texts/hamlet_2.txt', 'r', encoding="utf-8").read()
hamlet_3_text = open('texts/hamlet_3.txt', 'r', encoding="utf-8").read()

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
)


tokenizer.fit_on_texts([hamlet_1_text, hamlet_2_text, hamlet_3_text])
max_id = len(tokenizer.word_counts)
hamlet_1_encoded = np.array(tokenizer.texts_to_sequences(hamlet_1_text)) - 1
hamlet_2_encoded = np.array(tokenizer.texts_to_sequences(hamlet_2_text)) - 1
hamlet_3_encoded = np.array(tokenizer.texts_to_sequences(hamlet_3_text)) - 1

hamlet_1_encoded = hamlet_1_encoded.ravel()
hamlet_2_encoded = hamlet_2_encoded.ravel()
hamlet_3_encoded = hamlet_3_encoded.ravel()
hamlet_1_decoded = tokenizer.sequences_to_texts([hamlet_1_encoded + 1])

hamlet_1_dataset = tf.data.Dataset.from_tensor_slices(hamlet_1_encoded)
hamlet_2_dataset = tf.data.Dataset.from_tensor_slices(hamlet_2_encoded)
hamlet_3_dataset = tf.data.Dataset.from_tensor_slices(hamlet_3_encoded)

T = 100
window_length = T + 1

hamlet_1_dataset = hamlet_1_dataset.window(
    size=window_length, shift=1, drop_remainder=True
)
hamlet_2_dataset = hamlet_2_dataset.window(
    size=window_length, shift=1, drop_remainder=True
)
hamlet_3_dataset = hamlet_3_dataset.window(
    size=window_length, shift=1, drop_remainder=True
)

hamlet_1_dataset = hamlet_1_dataset.flat_map(lambda window: window.batch(window_length))
hamlet_2_dataset = hamlet_2_dataset.flat_map(lambda window: window.batch(window_length))
hamlet_3_dataset = hamlet_3_dataset.flat_map(lambda window: window.batch(window_length))

hamlet_conc = hamlet_1_dataset.concatenate(hamlet_2_dataset)
hamlet_dataset = hamlet_conc.concatenate(hamlet_3_dataset)

tf.random.set_seed(0)
# YOUR CODE

batch_size = 32
hamlet_dataset = hamlet_dataset.repeat()
hamlet_dataset = hamlet_dataset.shuffle(buffer_size=10000)
hamlet_dataset = hamlet_dataset.batch(batch_size, drop_remainder=True)

hamlet_dataset = hamlet_dataset.map(
    lambda window_batch: (window_batch[:, 0:100], window_batch[:, 1:101])
)

hamlet_dataset = hamlet_dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)

hamlet_dataset = hamlet_dataset.prefetch(buffer_size=1)
steps_per_epoch = int(
    ((len(hamlet_1_encoded) + len(hamlet_2_encoded) + len(hamlet_3_encoded)) - 3 * T)
    / batch_size
)


model = keras.models.Sequential(
    [
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax")),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)


history = model.fit(
    hamlet_dataset,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    callbacks=callback,
)

# model.save("hamlet_model_new.h5")
model.save("hamlet_model.h5")