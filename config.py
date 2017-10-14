# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum

class eOptimizer(Enum):
    Adam    = 1
    GD      = 2
    RMS     = 3
    
train_example_path  = 'D:/data/cdiscount/train_example.bson'
train_path          = 'D:/data/cdiscount/train.bson'
test_path           = 'D:/data/cdiscount/test.bson'
logger_path         = 'log.txt'

processed_train_dir = 'D:/data/cdiscount/processed/train/'
processed_test_dir  = 'D:/data/cdiscount/processed/test/'

log_dir             = 'log/'
checkpoint_dir      = 'checkpoint/'
output_dir          = 'output/'

key_product_id      = '_id'
key_category_id     = 'category_id'
key_imgs            = 'imgs'
key_picture         = 'picture'

train_ratio          = 0.8
data_dtype          = np.float32
label_dtype         = np.int64
dropout             = 0.5
learning_rate       = 0.001
optimizer           = eOptimizer.Adam
num_epoch           = 100
eval_step           = 5
batch_size          = 2
num_classes         = 10
