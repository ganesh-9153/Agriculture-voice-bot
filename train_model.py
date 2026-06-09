import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv('crop_data.csv')

X = data[['N', 'P', 'K', 'temperature', 'humidity']]
y = data['crop']

model = DecisionTreeClassifier()
model.fit(X, y)

pickle.dump(model, open('crop_model.pkl', 'wb'))

print("Model Trained Successfully")