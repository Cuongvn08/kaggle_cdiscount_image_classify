# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


## STEP0: do setting
class Settings(Enum):

    def __str__(self):
        return self.value
        
    
## STEP1: process data
def _process_data():
    print('\n\nSTEP1: processing data ...')

    
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
    