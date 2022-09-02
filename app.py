import sys
import numpy as np
from scipy import stats
from scipy.fft import fft
import pickle
import json
from classifier import predict



def handler(event, context):
    data = json.loads(event["body"])
    
    np_data_acc = np.array([np.fromstring(data["acc_x"], sep=" "), np.fromstring(data["acc_y"], sep=" "),
            np.fromstring(data["acc_z"], sep=" ")])
            
    np_data_gyr = np.array([np.fromstring(data["gyr_x"], sep=" "),
            np.fromstring(data["gyr_y"], sep=" "), np.fromstring(data["gyr_z"], sep=" ")])
    prediction = predict(np_data_acc.transpose(), np_data_gyr.transpose())
    print(prediction)
    return str({"Prediction": str(prediction)})   
  
      