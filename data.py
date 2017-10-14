# -*- coding: utf-8 -*-

import numpy as np
import random
import bson # conda install -c anaconda pymongo
from skimage.data import imread
import scipy.misc as misc
import io
import os
import shutil
import time
import config
import helper


class Data():
    def __init__(self, logger):
        self.logger         = logger
        
        self.train_data     = []
        self.train_label    = []

        self.eval_data      = []
        self.eval_label     = []

        self.test_data      = []
        self.test_label     = []

    def extract_train_data(self):
        helper.log(self.logger, '[data] Extracting and save train data into images ...')
        
        # Delete the current images    
        if os.path.exists(config.processed_train_dir):
            shutil.rmtree(config.processed_train_dir)
        os.makedirs(config.processed_train_dir)
        time.sleep(1)
    
        # load and save new train images
        train_bson = bson.decode_file_iter(open(config.train_path, 'rb'))           
        for c, d in enumerate(train_bson):
            product_id = d[config.key_product_id]
            category_id = d[config.key_category_id]
            helper.log(self.logger,'   Product, Category: {0}, {1}'.format(product_id, category_id))
                        
            for e, pic in enumerate(d[config.key_imgs]):
                picture = imread(io.BytesIO(pic[config.key_picture]))
                picture_path =  config.processed_train_dir + \
                                str(product_id)  + '_' + \
                                str(category_id) + '_' + \
                                str(e) +'.png'
                misc.imsave(picture_path, picture)

    def extract_test_data(self):
        helper.log(self.logger, '[data] Extracting and save test data into images ...')
        
        # Delete the current images    
        if os.path.exists(config.processed_test_dir):
            shutil.rmtree(config.processed_test_dir)
        os.makedirs(config.processed_test_dir)
        time.sleep(1)
    
        # load and save new train images
        test_bson = bson.decode_file_iter(open(config.test_path, 'rb'))           
        for c, d in enumerate(test_bson):
            product_id = d[config.key_product_id]
            helper.log(self.logger, '   Product:   {0}'.format(product_id))
            
            for e, pic in enumerate(d[config.key_imgs]):
                picture = imread(io.BytesIO(pic[config.key_picture]))
                picture_path =  config.processed_test_dir + \
                                str(product_id)  + '_' + \
                                str(e) +'.png'
                misc.imsave(picture_path, picture)
                
    def read_train_data(self):
        helper.log(self.logger, '[data] Reading train data into arrays ...')
        
        del self.train_data[:]
        del self.train_label[:]
        
        del self.eval_data[:]
        del self.eval_label[:]
        
        # Load data into arrays
        train_bson = bson.decode_file_iter(open(config.train_example_path, 'rb'))
        for c, d in enumerate(train_bson):
            product_id = d[config.key_product_id]
            category_id = d[config.key_category_id]
            helper.log(self.logger, '   Product:   {0}'.format(product_id))
                        
            for e, pic in enumerate(d[config.key_imgs]):
                picture = imread(io.BytesIO(pic[config.key_picture]))
                self.train_data.append(picture)
                self.train_label.append(category_id)
                        
        # Divide data into train and eval data
        num_data = len(self.train_data)
        num_train = np.int(num_data * config.train_ratio)
        #num_eval = num_data - num_train
        
        train_indices = random.sample(range(num_data), num_train)
        eval_indices = [i for i in range(num_data) if i not in train_indices]        

        self.eval_data = [self.train_data[i] for i in eval_indices]
        self.eval_label = [self.train_label[i] for i in eval_indices]
        
        self.train_data = [self.train_data[i] for i in train_indices]
        self.train_label = [self.train_label[i] for i in train_indices]

        self.train_data = np.asarray(self.train_data)
        self.train_label = np.asarray(self.train_label)
        self.eval_data = np.asarray(self.eval_data)
        self.eval_label = np.asarray(self.eval_label)
                
        helper.log(self.logger, '[data] Train data shape: {0}'.format(self.train_data.shape))
        helper.log(self.logger, '[data] Train label shape: {0}'.format(self.train_label.shape))          
        helper.log(self.logger, '[data] Eval data shape: {0}'.format(self.eval_data.shape))
        helper.log(self.logger, '[data] Eval label shape: {0}'.format(self.eval_label.shape))
        
    def read_test_data(self, data_path):
        #TBD
        pass

    def get_train_data(self):
        return self.train_data
    
    def get_train_label(self):
        return self.train_label
    
    def get_eval_data(self):
        return self.eval_data
    
    def get_eval_label(self):
        return self.eval_label
    
    def get_test_data(self):
        return self.test_data
    
    def get_test_label(self):
        return self.test_label
