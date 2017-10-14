import numpy as np
import tensorflow as tf
import config

## print and log
def log(logger, str):
    print(str)
    logger.write(str + '\n')
    logger.flush()

## get loss
# @param logit: logit from model [None, num_classes]
# @param label: class index [None]
def get_loss(logit, label):
    label = tf.to_int64(label)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        logits=logit,
                                                        labels=label,
                                                        name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy)

    return loss

## get optimizer
# @param learning_rate:
# @param optimizer: optimizer method
def get_optimizer(learning_rate, optimizer):
    if optimizer == config.eOptimizer.Adam:
        return tf.train.AdamOptimizer(learning_rate = learning_rate,
                                      beta1 = 0.9,
                                      beta2 = 0.999,
                                      epsilon = 1e-10,
                                      use_locking = False,
                                      name = 'Adam')
    elif optimizer == config.eOptimizer.GD:
        return tf.train.GradientDescentOptimizer(learning_rate = learning_rate,
                                                 use_locking = False,
                                                 name = 'GradientDescent')
    elif optimizer == config.eOptimizer.RMS:
        return tf.train.RMSPropOptimizer(learning_rate = learning_rate,
                                         decay = 0.9,
                                         momentum = 0.0,
                                         epsilon = 1e-10,
                                         use_locking = False,
                                         centered = False,
                                         name = 'RMSProp')
    else:
        assert '[train] optimizer error'
        
## get batches
# @param data_len: data length (number of examples)
# @param batch_size: batch size (number of examples per training run)
# return: zip of batch starts and ends
def get_batches(data_len, batch_size):
    batch_starts = range(0, data_len, batch_size)
    batch_ends = [batch_start + batch_size for batch_start in batch_starts]
    return zip(batch_starts, batch_ends)
    
## get confusion matrix    
def get_confusion_matrix(pred, label):
    confusion_matrix = np.zeros((config.num_classes, config.num_classes), dtype=np.int)

    for (pred_idx, label_idx) in zip(pred, label):
        confusion_matrix[pred_idx, label_idx] += 1

    return confusion_matrix

## get accuracy
def get_accuracy(confusion_matrix):
    sum = np.sum(confusion_matrix)
    true_pred = np.sum(confusion_matrix[i][i] for i in range(len(confusion_matrix[0])))
    return true_pred/sum

## get precision
def get_precision(confusion_matrix):
    # do nothing
    pass

## get recall
def get_recall(confusion_matrix):
    # do nothing
    pass

## get f1score
def get_F1score(confusion_matrix):
    # do nothing
    pass