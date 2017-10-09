# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import io
import bson
from skimage.data import imread
import pandas as pd


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


## STEP0: do setting
class Settings(Enum):
    global train_example_path
    global train_path
    global test_path
    
    train_example_path = 'E:/data/kaggle/cdiscount/train_example.bson'
    train_path = 'E:/data/kaggle/cdiscount/train.bson'
    test_path = 'E:/data/kaggle/cdiscount/test.bson'
    
    def __str__(self):
        return self.value
        
    
## STEP1: process data
def _process_data():
    print('\n\nSTEP1: processing data ...')
    
    data = bson.decode_file_iter(open(train_example_path, 'rb'))
    
    prod_to_category = dict()
    
    for c, d in enumerate(data):
        product_id = d['_id']
        category_id = d['category_id'] # This won't be in Test data
        prod_to_category[product_id] = category_id
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            # do something with the picture, etc
    
    prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
    prod_to_category.index.name = '_id'
    prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

    print(prod_to_category.head())
    
    plt.imshow(picture)
    
## STEP2: build model
def _build_model():
    print('\n\nSTEP2: building model ...')
    

## STEP3: train    
def _train():
    print('\n\nSTEP3: training ...')
    
    
## STEP4: predict
def _predict():
    print('\n\nSTEP4: predicting ...')
    
    
## STEP5: generate submission    
def _generate_submission():
    print('\n\nSTEP5: generating submission ...')


## main
def main():
    _process_data()
    _build_model()
    _train()
    _predict()
    _generate_submission()
    

if __name__ == "__main__":
    main()
    print('\n\n\nThe end.')
    