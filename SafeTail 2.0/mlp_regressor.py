import numpy as np
import pickle

# Load the trained MLP model
with open("mlp_model.pkl", "rb") as model_file:
    mlp_regressor = pickle.load(model_file)

# Load the StandardScaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

def predict(inp):
    
    if not isinstance(inp, list) or len(inp) != 2:
        raise ValueError("Input should be a list with two elements: [metadata, feature_array].")
    
    feature_array = inp[1]
    print("Feature array shape:", feature_array.shape)  # Debugging line

    if not isinstance(feature_array, np.ndarray) or feature_array.shape[1] != 110:
        raise ValueError("Feature array must be a NumPy array with shape (1, 110).")

    # Reshape inp[0] to (1, 1) so it can be stacked properly
    metadata_feature = np.array([[inp[0]]])

    # Stack the metadata_feature and feature_array horizontally
    X_test = np.hstack((metadata_feature, feature_array))  # Shape becomes (1, 111)

    # Transform input using the scaler
    X_scaled = scaler.transform(X_test)

    # Make prediction
    pred = mlp_regressor.predict(X_scaled)[0]

    return pred

""" max_mem = [828.944,828.816,829.116,834.780,853.148]
cpu = [779,740,531,399,319]


def predict(inp):
    print("inp" , inp)  
    mlp_regr_model = open('yolo_thread_mlp_regressor_model.pkl', 'rb')
    mlp_regressor = pickle.load(mlp_regr_model)
    with open('yolo_thread_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    print("----------------------------------------")
    print(f"Type of inp[1]: {type(inp[1])}")
    print("----------------------------------------")
    print(f"Shape of inp[1]: {np.shape(inp[1])}")
    print("----------------------------------------")

    #X_test = np.array([[inp[0],inp[1]]])
    X_test = np.hstack(([inp[0]], inp[1].flatten()))
    X_test = np.array(X_test)
    print(f"X_test shape: {X_test.shape}")
    print(f"Scaler was fitted on {scaler.n_features_in_} features")
    st_dev = [2.7965017993200005, 6.393525858045778, 5.515155335473335, 7.100407896557774, 6.909932092510028, 7.124802705029803, 7.972889920066626, 9.872241111272556, 8.936860060306417, 12.450634470736823, 12.444299205142089, 11.830614817831743, 14.73346686870066, 17.763701566033472, 14.910366835514143, 19.22145836272576, 17.4644685302588, 19.87528975084137, 23.439168758290045, 20.457122880591005]
    pred = mlp_regressor.predict(scaler.transform(X_test))[0]
    return (pred + np.random.normal(0 , st_dev[int(inp[0])-1] ,1))/1000

 """