import tensorflow as tf
from sklearn.metrics import roc_auc_score

def avg_auroc_metric (y, y_hat):
    """Return a function that calculates the Area Under the ROC in order to use it as metric.
    y.shape : (n, N_LABELS)
    y_hat.shape : (n, N_LABELS)"""
    return tf.py_function(avg_auroc, (y, y_hat), tf.double)

def avg_auroc (y, y_hat):
    """Calculates the label-wise average Area Under the ROC across multiple labels
    y.shape : (n, N_LABELS)
    y_hat.shape : (n, N_LABELS)"""
    accumulator = 0
    try:
        for label in range(y.shape[1]):
            accumulator += roc_auc_score(y[:,label], y_hat[:,label])
        return accumulator / y.shape[1]
    except:
        return float("nan")

class AvgAurocCallback (tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.roc_train = []
        self.roc_val = []
        super().__init__()
    def on_epoch_end (self, epoch, logs=None):
        # measure train set average auroc
        train_y_hat = self.model.predict(self.X_train)
        train_avg_auroc = avg_auroc(self.y_train, train_y_hat)
        # and val set average auroc
        val_y_hat = self.model.predict(self.X_val)
        val_avg_auroc = avg_auroc(self.y_val, val_y_hat)
        self.roc_train.append(train_avg_auroc)
        self.roc_val.append(val_avg_auroc)
        print("\nTrain avg_auroc: {:.3f}, Val avg_auroc: {:.3f}".format(train_avg_auroc, val_avg_auroc))
