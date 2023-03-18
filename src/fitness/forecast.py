from fitness.base_ff_classes.base_ff import base_ff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from math import sqrt
from random import random
from algorithm.parameters import params
from stats.stats import stats
from utilities.stats.trackers import cache
from warnings import catch_warnings
from warnings import filterwarnings


LM_CLASSES = {
    'ar': AutoReg,
    'arima': ARIMA,
    'sarima': SARIMAX,
    'ses': SimpleExpSmoothing,
    'hwes': ExponentialSmoothing,
}

NM_CLASSES = {
    'mlp': MLPRegressor,
    'svr': SVR,
}


class forecast(base_ff):


    def __init__(self):
        super().__init__()
        data = pd.read_csv(params['DATASET_URL'], header=None)
        X = data[0].values
        split_point = int(len(X) * float(params['SPLIT_PROPORTION']))
        train, test = X[0:split_point], X[split_point:len(X)]
        self.split_point = split_point
        self.train = train
        self.test = test
        self.data = X

    
    def build_linear_model(self, phenotype):

        classe = None
        class_params = {}
        fit_params = {}

        blocks = phenotype.split(';')

        for block in blocks:

            if 'lm:' in block:
                
                parts = block.split(' ')
                func = parts[1]

                classe = LM_CLASSES[func]
                
                if func == 'ar':
                    class_params = {
                        'lags': 1
                    }
                elif func == 'arima':
                    order = eval(parts[2])
                    class_params = {
                        'order': order
                    }
                elif func == 'sarima':
                    order = eval(parts[2])
                    seasonal_order = eval(parts[3])
                    trend = parts[4]
                    class_params = {
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'trend': trend,
                    }
                    fit_params = {
                        'disp': False
                    }

        return classe, class_params, fit_params
        

    def build_nonlinear_model(self, phenotype):
        
        blocks = phenotype.split(';')

        classe = None
        class_params = {}

        for block in blocks:

            if 'nm:' in block:
                
                parts = block.strip().split(' ')
                func = parts[1]

                classe = NM_CLASSES[func]
                
                if func == 'mlp':
                    aux = eval(parts[2])
                    class_params = {
                        'hidden_layer_sizes': aux[0],
                        'activation': aux[1],
                        'solver': aux[2],
                        'learning_rate': aux[3],
                    }
                elif func == 'svr':
                    aux = eval(parts[2])
                    class_params = {
                        'kernel': aux[0],
                        'max_iter': -1, 
                        'shrinking': True,
                        'C': aux[1],
                        'gamma': aux[2],
                        'epsilon': aux[3],
                    }
        
        return classe, class_params

    
    def convert_series_to_nm_dataset(self, series):
  
        df = pd.DataFrame(series)
        df = pd.concat([df.shift(1), df], axis=1)
        df.fillna(-1, inplace=True)

        values = df.values
        X, y = values, values[:, 0]

        return X, y


    def compute_lm(self, classe, class_params, fit_params):
        
        model = classe(self.train, **class_params)
        model_fit = model.fit(**fit_params)

        predictions = model_fit.forecast(len(self.test))

        return mean_squared_error(self.test, predictions)

    
    def compute_nm(self, classe, class_params):

        X, y = self.convert_series_to_nm_dataset(self.data)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=len(X) - self.split_point)
            
        model = classe(**class_params)
        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)

        return mean_squared_error(Y_test, predictions)

    
    def compute_lm_and_nm(self, lm_class, lm_class_params, lm_fit_params, nm_class, nm_class_params):

        model = lm_class(self.train, **lm_class_params)
        model_fit = model.fit(**lm_fit_params)

        resid_train = model_fit.resid

        predictions = model_fit.forecast(len(self.test))
        resid_val = self.test - predictions

        resid = np.concatenate([resid_train, resid_val])

        X, y = self.convert_series_to_nm_dataset(resid)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=len(X) - self.split_point)
            
        model = nm_class(**nm_class_params)
        model.fit(X_train, Y_train)

        resid_predictions = model.predict(X_test)

        return mean_squared_error(Y_test, predictions + resid_predictions)



    def evaluate(self, ind, **kwargs):
        
        print('GENERATION:', stats['gen'])
        print('PHENOTYPE:', ind.phenotype)
        
        error = float('inf')

        try:

            if ind.phenotype in cache:

                error = cache[ind.phenotype]
            
            else:
            
                lm_class, lm_class_params, lm_fit_params = self.build_linear_model(ind.phenotype)
                nm_class, nm_class_params = self.build_nonlinear_model(ind.phenotype)

                if lm_class and not nm_class:
                    error = self.compute_lm(lm_class, lm_class_params, lm_fit_params)
                elif nm_class and not lm_class:
                    error = self.compute_nm(nm_class, nm_class_params)
                else:
                    error = self.compute_lm_and_nm(lm_class, lm_class_params, lm_fit_params, nm_class, nm_class_params)

        except ValueError as ex:
            print(ex)

        print('FITNESS:', error)

        return error