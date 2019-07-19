import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

MAX_WORDS = 20000
VALIDATION_SPLIT = .2
MAX_SEQ_LENGTH = 1000
embedding_dims = [50, 100, 200, 300]
max_length = 16
oov_tok = "<OOV>"
corpus = []
DATA_PATH = './data/'
num_epochs = 10

print('Reading training data')
train = pd.read_csv(DATA_PATH + "train.csv")
labels = train[["toxic", "severe_toxic", "obscene",
                "threat", "insult", "identity_hate"]].values.tolist()
labels = np.array(labels)
sentences = train["comment_text"].values.tolist()

# word tokenization
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("Words in training set: " + str(len(word_index)))

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)

# split the data into a training set and a validation set
indices = np.arange(padded.shape[0])
np.random.shuffle(indices)
padded = padded[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * padded.shape[0])

x_train = padded[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = padded[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

for embedding_dim in embedding_dims:

    print(f'********Running experiment for {embedding_dim}-D*********')
    glove_path = DATA_PATH + f'glove.6B/glove.6B.{embedding_dim}d.txt'

    # load pretrained embedding matrix (implemented as an index for memory efficiency)
    embeddings_index = {}
    with open(glove_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # prepare embedding matrix
    num_words = min(MAX_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i > MAX_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQ_LENGTH,
                                trainable=False)

    from avg_auroc import AvgAurocCallback
    avg_auroc_callback = AvgAurocCallback(x_train, y_train, x_val, y_val)

    model = tf.keras.Sequential([
                                embedding_layer,
                                tf.keras.layers.Dropout(rate=0.2),
                                tf.keras.layers.Conv1D(64, 5,
                                                       activation='relu'),
                                tf.keras.layers.MaxPooling1D(pool_size=4),
                                tf.keras.layers.CuDNNLSTM(64),
                                tf.keras.layers.Dense(6, activation='sigmoid')
                                ])

    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    history = model.fit(x_train, y_train, epochs=num_epochs, verbose=1,
                        batch_size=128, validation_data=(x_val, y_val),
                        callbacks=[avg_auroc_callback]
                        )

    history.history['roc_train'] = avg_auroc_callback.roc_train
    history.history['roc_val'] = avg_auroc_callback.roc_val

    hist_file = open(f"baseline_{embedding_dim}D_history", "wb")
    pickle.dump(history.history, hist_file)
    hist_file.close()
