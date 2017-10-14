# -*- coding: utf-8 -*-
import os
import shutil
import time
import datetime
import numpy as np
import tensorflow as tf
import config
import data
import helper
import model

# Create log file
with tf.name_scope('logger'):
    if os.path.exists(config.logger_path):
        os.remove(config.logger_path)
    
    logger = open(config.logger_path, 'w')
    helper.log(logger, 'CDISCOUNT')
    
# Create session
with tf.name_scope('session'):
    sess = tf.Session()

# Load data
with tf.name_scope('data'):
    data = data.Data(logger)

    data.read_train_data()

    train_data  = data.get_train_data()
    train_label = data.get_train_label()

    eval_data  = data.get_eval_data()
    eval_label = data.get_eval_label()

# Create placeholders
with tf.name_scope('placeholder'):
    data = tf.placeholder(config.data_dtype, shape=[None, 180, 180, 3], name='data')
    label = tf.placeholder(config.label_dtype, shape=[None], name='label')

    tf.add_to_collection('data', data)
    tf.add_to_collection('label', label)

    helper.log(logger, '[train] Shape of data placeholder {0}'.format(data.get_shape()))
    helper.log(logger, '[train] Shape of label placeholder {0}'.format(label.get_shape()))
    
# Create model
with tf.name_scope('model'):
    model = model.Model(logger)

# Get train opt
with tf.name_scope('train'):
    train_logit = model.logit(data, True, config.dropout)
    train_cost = helper.get_loss(train_logit, label)
    train_opt = helper.get_optimizer(config.learning_rate, config.optimizer).minimize(train_cost)

    train_pred = tf.argmax(tf.nn.softmax(train_logit), axis=1, name='train_pred')
    train_equal = tf.equal(train_pred, label)
    train_acc = tf.reduce_mean(tf.cast(train_equal, tf.float32))

    train_summary_list = []
    train_summary_list.append(tf.summary.scalar('train_cost', train_cost))
    train_summary_list.append(tf.summary.scalar('train_acc', train_acc))
    train_summary_merge = tf.summary.merge(train_summary_list)

# get eval opt
with tf.name_scope('eval'):
    tf.get_variable_scope().reuse_variables()
    eval_logit = model.logit(data, False)
    eval_cost = helper.get_loss(eval_logit, label)

    eval_pred = tf.argmax(tf.nn.softmax(eval_logit), axis=1, name='pred')
    eval_equal = tf.equal(eval_pred, label)
    eval_acc = tf.reduce_mean(tf.cast(eval_equal, tf.float32))

    eval_summary_list = []
    eval_summary_list.append(tf.summary.scalar('eval_cost', eval_cost))
    eval_summary_list.append(tf.summary.scalar('eval_acc', eval_acc))
    eval_summary_merge = tf.summary.merge(eval_summary_list)

    tf.add_to_collection('pred', eval_pred)

# Initialize variables
with tf.name_scope('initialize_variables'):
    sess.run(tf.global_variables_initializer())

# Create summary
with tf.name_scope('summary'):
    if os.path.exists(config.log_dir) is True:
        shutil.rmtree(config.log_dir)
    os.makedirs(config.log_dir)

    summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

# create saver
with tf.name_scope('saver'):
    if os.path.exists(config.checkpoint_dir) is True:
        shutil.rmtree(config.checkpoint_dir)
    os.makedirs(config.checkpoint_dir)

    saver = tf.train.Saver(max_to_keep = None)

'''
# train
train_batches = get_batches(train_data.shape[0], cfig[eKey.batch_size])
for start, end in train_batches:
    print_log(logger, '[train] train_batches: start={0}, end={1}'.format(start, end))
    
logger.write('\n')
'''

# train
with tf.name_scope('train'):
    with tf.device('/cpu:%d' % 0):
    #with tf.device('/gpu:%d' % 0):
        for step in range(config.num_epoch):
            # train
            start_time = time.time()
            train_fetches = [train_logit, train_cost, train_opt, train_summary_merge]

            train_costs = []
            train_batches = helper.get_batches(train_data.shape[0], config.batch_size)
            for start, end in train_batches:
                feed_dict = {}
                feed_dict[data] = train_data[start:end]
                feed_dict[label] = train_label[start:end]
                [_, tCost, _, tSummary] = sess.run(train_fetches, feed_dict)
                train_costs.append(tCost)

            # eval
            if step % config.eval_step == 0:
                eval_fetches = [eval_logit, eval_cost, eval_pred, eval_equal, eval_acc, eval_summary_merge]

                feed_dict = {}
                feed_dict[data] = eval_data
                feed_dict[label] = eval_label
                [_, eCost, ePred, eEqual, eAcc, eSummary] = sess.run(eval_fetches, feed_dict)

                # log and print
                elapsed_time = time.time() - start_time
                date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                str = '[train] {0} step {1:04}: tCost = {2:0.5}; eCost = {3:0.5}; eAcc = {4:0.5}; time = {5:0.5}(s);'.\
                    format(date_time, step, np.mean(train_costs), eCost, eAcc, elapsed_time)
                helper.log(logger, str)

                # get confusion matrix
                confusion_matrix = helper.get_confusion_matrix(ePred, eval_label)
                if eAcc > 0.99:
                    print(confusion_matrix)

                # save summaries
                summary_writer.add_summary(tSummary, step)
                summary_writer.add_summary(eSummary, step)
                summary_writer.flush()

                # save checkpoint
                saver.save(sess, config.checkpoint_dir + 'chk_step_%d.ckpt'%step)

                
                # show 3 random predictions
                #idx = random.sample(range(eval_data.shape[0]), 3)

                #plt.subplot(131)
                #plt.imshow(np.squeeze(eval_data[idx[0]]), cmap='gray')
                #plt.title('label = {0}; pred = {1}'.format(eval_label[idx[0]], ePred[idx[0]]))

                #plt.subplot(132)
                #plt.imshow(np.squeeze(eval_data[idx[1]]), cmap='gray')
                #plt.title('label = {0}; pred = {1}'.format(eval_label[idx[1]], ePred[idx[1]]))

                #plt.subplot(133)
                #plt.imshow(np.squeeze(eval_data[idx[2]]), cmap='gray')
                #plt.title('label = {0}; pred = {1}'.format(eval_label[idx[2]], ePred[idx[2]]))

                #plt.pause(0.1)
                
    summary_writer.close()

# end
helper.log(logger, 'The end.')
logger.close()
