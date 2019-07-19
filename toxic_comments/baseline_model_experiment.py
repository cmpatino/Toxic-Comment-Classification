import numpy as np
import pandas as pd
import pickle

# tf-related
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant
from avg_auroc import AvgAurocCallback
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


DATA_PATH = './data/'

def run_experiments():

    filter_sizes = [3, 5]
    dropout_values = [0.2, 0.5]
    max_words_values = [10000, 100000]
    max_seq_length_values = [150, 1000]

    print('Reading training data')
    train = pd.read_csv(DATA_PATH + "train.csv")
    labels = train[["toxic", "severe_toxic", "obscene",
                    "threat", "insult", "identity_hate"]].values.tolist()
    labels = np.array(labels)
    sentences = train["comment_text"].values.tolist()

    i = 0
    for filter_size in filter_sizes:
        for dropout in dropout_values:
            for max_words in max_words_values:
                for max_seq_length in max_seq_length_values:

                    i += 1
                    print(f'********Running experiment for {filter_size}_{int(dropout*10)}_{max_words}_{max_seq_length}*********')
                    print(f"Experiment {i}/16")
                    baseline_model(sentences, labels, filter_size, dropout, max_seq_length, max_words)


def baseline_model(sentences, labels, filter_size, dropout, max_seq_length, max_words=100000, embedding_dim=200, num_epochs=10):

    validation_split = .2
    num_epochs = 10

    # word tokenization
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print("Words in training set: " + str(len(word_index)))

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_seq_length)

    # split the data into a training set and a validation set
    indices = np.arange(padded.shape[0])
    np.random.shuffle(indices)
    padded = padded[indices]
    labels = labels[indices]
    num_validation_samples = int(validation_split * padded.shape[0])

    x_train = padded[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = padded[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

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
    num_words = min(max_words, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i > max_words:
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
                                input_length=max_seq_length,
                                trainable=False)

    avg_auroc_callback = AvgAurocCallback(x_train, y_train, x_val, y_val)

    model = tf.keras.Sequential([
                                embedding_layer,
                                tf.keras.layers.Dropout(rate=dropout),
                                tf.keras.layers.Conv1D(64, filter_size,
                                                       activation='relu'),
                                tf.keras.layers.MaxPooling1D(pool_size=4),
                                tf.keras.layers.CuDNNLSTM(64),
                                tf.keras.layers.Dense(6, activation='sigmoid')
                                ])

    model.compile(loss='binary_crossentropy', optimizer='adam')

    saved_model_path = './baseline_exp/saved_models/' + f"baseline_{filter_size}_{int(dropout*10)}_{max_words}_{max_seq_length}.h5"
    checkpoint = ModelCheckpoint(saved_model_path, monitor='val_loss',
                                 save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                                  mode='min', min_lr=0.00001)

    history = model.fit(x_train, y_train, epochs=num_epochs, verbose=1,
                        batch_size=256, validation_data=(x_val, y_val),
                        callbacks=[avg_auroc_callback, checkpoint, reduce_lr]
                        )

    history.history['roc_train'] = avg_auroc_callback.roc_train
    history.history['roc_val'] = avg_auroc_callback.roc_val

    test = pd.read_csv(DATA_PATH + "test.csv")
    test_sentences = test.pop("comment_text").values.tolist()

    # make the sequences uniform to pass them to a network
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences, maxlen=max_seq_length)

    # make predictions
    model.load_weights(filepath=saved_model_path)
    y_test_hat = model.predict(test_padded)

    # write submission file
    y_test_hat = pd.DataFrame(data=y_test_hat, columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    test = test.join(y_test_hat)
    test.to_csv('./baseline_exp/submissions/' + f"baseline_{filter_size}_{int(dropout*10)}_{max_words}_{max_seq_length}_submission.csv", index=False)

    hist_file = open('./baseline_exp/histories/' + f"baseline_{filter_size}_{int(dropout*10)}_{max_words}_{max_seq_length}_history", "wb")
    pickle.dump(history.history, hist_file)
    hist_file.close()


if __name__ == '__main__':

    run_experiments()
