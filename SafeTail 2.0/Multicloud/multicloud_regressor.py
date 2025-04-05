import numpy as np
import pandas as pd 
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder



#loading dataset 
df= pd.read_csv('/home/user/khushi/SafeTail 2.0/datasets/multi_cloud_service_dataset.csv')

encoder = LabelEncoder()
df['Service_Type'] = encoder.fit_transform(df['Service_Type'])

print(df['service type'])
feature_cols = ['Service_Type', 'CPU_Utilization (%)', 'Memory_Usage (MB)', 
                'Storage_Usage (GB)', 'Network_Bandwidth (Mbps)', 'Throughput (Requests/sec)']
target_cols = ['Service_Latency (ms)', 'QoS_Score']

X = df[feature_cols].values
y = df[target_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler.feature_names = feature_cols

mlp_regressor = MLPRegressor(hidden_layer_sizes=(64, 128, 64), activation='relu', max_iter=2000, random_state=42)
mlp_regressor.fit(X_train, y_train)

y_pred = mlp_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print("Mean Squared Error for each server:", mse)

with open('Multicloud/multicloud_model.pkl', 'wb') as model_file:
    pickle.dump(mlp_regressor, model_file)

with open('Multicloud/multicloud_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")