# 1. Load  the training and testing data
# 2. Scale the trining data
# 3. save scaled data into processed folder

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_and_split_data

X_train,X_test,y_train,y_test=load_and_split_data()
scalar=StandardScaler()
X_train_scaled=scalar.fit_transform(X_train)
X_test_scaled=scalar.transform(X_test)
pd.DataFrame(X_train_scaled).to_csv(r'../data/process/X_train.csv',index=False)
pd.DataFrame(X_test_scaled).to_csv(r'../data/process/X_test.csv',index=False)
pd.DataFrame(y_train).to_csv(r'../data/process/y_train.csv',index=False)
pd.DataFrame(y_test).to_csv(r'../data/process/y_test.csv',index=False)

with open(r'../artifacts/scalar.pkl','wb') as f:
    pickle.dump(scalar,f)

print("Successfully saves the scaled data and scalar object")