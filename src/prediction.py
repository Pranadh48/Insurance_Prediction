# 1. lead scalar and model from artifact folder
# 2. create a function to predict the insurance premium

import pickle
import numpy as np
from pathlib import Path

class Insurance_Prediction:
    def __init__(self):
        # artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"

        # with open(artifacts_dir / "scalar.pkl", "rb") as f:
        #     self.scalar = pickle.load(f)
        # with open(artifacts_dir / "model.pkl", "rb") as f:
        #     self.model = pickle.load(f)
        

        self.scalar = pickle.load(open("artifacts/scalar.pkl", "rb"))
        
        self.model = pickle.load(open("artifacts/model.pkl", "rb"))
        
    
    def prediction_model(self,Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        input = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])
        scaled_input = self.scalar.transform(input)
        result = self.model.predict(scaled_input)
        return result[0]