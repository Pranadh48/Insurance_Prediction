# 1. lead processed data from process folder
# 2. create model and train data
# 3. save model in artifact folder

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


X_train=pd.read_csv(r'../data/process/X_train.csv')
X_test=pd.read_csv(r'../data/process/X_test.csv')
y_train=pd.read_csv(r'../data/process/y_train.csv')
y_test=pd.read_csv(r'../data/process/y_test.csv')

model = LinearRegression()
model.fit(X_train, y_train)

with open(r'../artifacts/model.pkl','wb') as f:
    pickle.dump(model,f)
