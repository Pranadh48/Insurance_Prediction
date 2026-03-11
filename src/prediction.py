# 1. lead scalar and model from artifact folder
# 2. create a function to predict the insurance premium

import pickle
import numpy as np

class Insurance_Prediction:
    def __init__(self):
        with open('C:\\Users\\Pranadh\\OneDrive\\Desktop\\Tek_Projects\\Insurance_Prediction\\artifacts\\scalar.pkl','rb') as f:
            self.scalar=pickle.load(f)
        with open('C:\\Users\\Pranadh\\OneDrive\\Desktop\\Tek_Projects\\Insurance_Prediction\\artifacts\\model.pkl','rb') as f:
            self.model=pickle.load(f)
    
    def prediction_model(self,Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        input = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])
        scaled_input = self.scalar.transform(input)
        result = self.model.predict(scaled_input)
        return result[0]