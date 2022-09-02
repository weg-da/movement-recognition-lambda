import pickle
import numpy as np
from scipy import stats
from features.feature_extraction_funcs import find_main_freq, spectral_entropy, zcr
from scipy import stats
from scipy.fft import fft


# Loading data


# loading mean and std for normalization
features_mean_time = np.load('/var/task//features/features_mean.npy', allow_pickle=True) 
features_std_time = np.load('/var/task//features/features_std.npy', allow_pickle=True) 
features_mean_freq = np.load('/var/task//features/features_mean_freq.npy', allow_pickle=True) 
features_std_freq = np.load('/var/task//features/features_std_freq.npy', allow_pickle=True)

with open('/var/task/models/model_linear.sav', 'rb') as f:
    model_linear = pickle.load(f)
with open('/var/task/models/model_sigmoid.sav', 'rb') as f:
    model_sigmoid = pickle.load(f)



# mat is the matrix for one person
def predict(mat_acc, mat_gyr):
    

    for sensor in range (1,3,1):
        if (sensor == 1):
          mat = mat_acc
        else:
          mat = mat_gyr
    # for mat in [mat_acc, mat_gyr]:
       
        x = mat[:,0]                
        y = mat[:,1]                                
        z = mat[:,2]               
        
        if (np.max(x) - np.min(x) != 0):
          x = (x - np.min(x))/(np.max(x) - np.min(x))
        if (np.max(y) - np.min(y) != 0):
          y = (y - np.min(y))/(np.max(y) - np.min(y))
        if (np.max(z) - np.min(z) != 0):
          z = (z - np.min(z))/(np.max(z) - np.min(z))
        
        
        
        ## Calculating the features
        # Median
        x_med = np.array(np.median(x))
        y_med = np.array(np.median(y))
        z_med = np.array( np.median(z))
        
        # Standard deviation
        x_std = np.array(np.std(x, dtype=np.float64))
        y_std = np.array(np.std(y, dtype=np.float64))
        z_std = np.array(np.std(z, dtype=np.float64))
        
        # Variance
        x_var = np.array(np.var(x))
        y_var = np.array(np.var(y))
        z_var = np.array(np.var(z))
        
        # Percentile 25th
        x_p25 = np.array(np.percentile(x, 25))
        y_p25 = np.array(np.percentile(y, 25))
        z_p25 = np.array(np.percentile(z, 25))
        
        # Percentile 75th
        x_p75 = np.array(np.percentile(x, 75))
        y_p75 = np.array(np.percentile(y, 75))
        z_p75 = np.array(np.percentile(z, 75))
        
        
        # Root mean square
        x_rms = np.array(np.sqrt(np.mean(np.square(x))))
        y_rms = np.array(np.sqrt(np.mean(np.square(y))))
        z_rms = np.array(np.sqrt(np.mean(np.square(z))))
        
        # Zero crossing rate

        x_zcr = np.array(zcr(x))
        y_zcr = np.array(zcr(y))
        z_zcr = np.array(zcr(z))
        
        # Person correlation coefficient
        xy_pers = np.array(np.corrcoef(x,y)[0][1]) 
        xz_pers = np.array(np.corrcoef(x,z)[0][1]) 
        yz_pers = np.array(np.corrcoef(y,z)[0][1]) 
    
        
        # Cross correlation (maximum)
        xy_xcr_max = np.array(np.correlate(x, y))[0]
        xz_xcr_max = np.array(np.correlate(x, z))[0]
        yz_xcr_max = np.array(np.correlate(y, z))[0]
        
        # Cross correlation (std)
        xy_xcr_std = np.array(np.std(np.correlate(x, y, 'full')))
        xz_xcr_std = np.array(np.std(np.correlate(x, z, 'full')))
        yz_xcr_std = np.array(np.std(np.correlate(y, z, 'full')))


        # Frequency Features

        x_spec = fft(x)
        y_spec = fft(y)
        z_spec = fft(z)
        
        N = x.size
        f = np.arange(0, 200/2, 200/N)                # frequency vector   
        x_pxx = (2/(N^2))*abs(x_spec)**2
        y_pxx = (2/(N^2))*abs(y_spec)**2
        z_pxx = (2/(N^2))*abs(z_spec)**2

        x_pxx = x_pxx/sum(x_pxx)
        y_pxx = y_pxx/sum(y_pxx)
        z_pxx = z_pxx/sum(z_pxx)

        # DC component
        x_dc_comp = x_pxx[0]
        y_dc_comp = y_pxx[0]
        z_dc_comp = z_pxx[0]
        
        # Main frequency & main frequency (amplitude)
        x_main_freq = find_main_freq(x_pxx[1:N//2], f[1:])[0]
        y_main_freq = find_main_freq(y_pxx[1:N//2], f[1:])[0]
        z_main_freq = find_main_freq(z_pxx[1:N//2], f[1:])[0]
       
        x_main_freq_amp = np.max(x_pxx[1:N//2])
        y_main_freq_amp = np.max(y_pxx[1:N//2])
        z_main_freq_amp = np.max(z_pxx[1:N//2])


        # spectral entropy
        x_spec_entro = spectral_entropy(x_pxx[1:N//2])
        y_spec_entro = spectral_entropy(y_pxx[1:N//2])
        z_spec_entro = spectral_entropy(z_pxx[1:N//2])


        ## Calculating the features
        # Median
        x_med_f = np.median(x_pxx)
        y_med_f = np.median(y_pxx)
        z_med_f = np.median(z_pxx)
        
        # Standard deviation
        x_std_f = np.std(x, dtype=np.float64)
        y_std_f = np.std(y, dtype=np.float64)
        z_std_f = np.std(z, dtype=np.float64)
        

        # Percentile 25th
        x_p25_f = np.percentile(x_pxx, 25)
        y_p25_f = np.percentile(y_pxx, 25)
        z_p25_f = np.percentile(z_pxx, 25)
        
        # Percentile 75th
        x_p75_f = np.percentile(x_pxx, 75)
        y_p75_f = np.percentile(y_pxx, 75)
        z_p75_f = np.percentile(z_pxx, 75)
        
        
        # Person correlation coefficient
        xy_pers_f = np.corrcoef(x_pxx,y_pxx)[0][1]
        xz_pers_f = np.corrcoef(x_pxx,z_pxx)[0][1]
        yz_pers_f = np.corrcoef(y_pxx,z_pxx)[0][1]
        
        # Cross correlation (maximum)
        xy_xcr_max_f = np.correlate(x_pxx, y_pxx)
        xz_xcr_max_f = np.correlate(x_pxx, z_pxx)
        yz_xcr_max_f = np.correlate(y_pxx, z_pxx)
        
        # Cross correlation (std)
        xy_xcr_std_f = np.std(np.correlate(x_pxx, y_pxx, 'full'))
        xz_xcr_std_f = np.std(np.correlate(x_pxx, z_pxx, 'full'))
        yz_xcr_std_f = np.std(np.correlate(y_pxx, z_pxx, 'full'))

        time_features = np.array([x_med, y_med, z_med, x_std, y_std, z_std, x_var, y_var, z_var, x_p25, y_p25, z_p25, \
                                 x_p75, y_p75, z_p75, x_rms, y_rms, z_rms, x_zcr, y_zcr, z_zcr, xy_pers, xz_pers, yz_pers, \
                                 xy_xcr_max, xz_xcr_max, yz_xcr_max, xy_xcr_std, xz_xcr_std, yz_xcr_std ],
                                dtype=object)
        freq_features = np.array([x_dc_comp, y_dc_comp, z_dc_comp, \
                         x_main_freq, y_main_freq, z_main_freq, \
                         x_main_freq_amp, y_main_freq_amp, z_main_freq_amp, \
                         x_spec_entro, y_spec_entro, z_spec_entro, \
                         x_med_f, y_med_f, z_med_f, \
                         x_std_f, y_std_f, z_std_f, \
                         x_p25_f, y_p25_f, z_p25_f, \
                         x_p75_f, y_p75_f, z_p75_f, \
                         xy_pers_f, xz_pers_f, yz_pers_f, \
                         xy_xcr_max_f, xz_xcr_max_f, yz_xcr_max_f, \
                         xy_xcr_std_f, xz_xcr_std_f, yz_xcr_std_f ], dtype=object)
        
        
        if sensor == 1:
            # x_min, y_min, z_min, x_max, y_max, z_max, x_range, y_range, z_range, \
            acc_time = time_features
            acc_freq = freq_features
                
                
                
            
        if sensor == 2:
            # x_min, y_min, z_min, x_max, y_max, z_max, x_range, y_range, z_range, \
            gyr_time = time_features
            gyr_freq = freq_features
                
    
    
    sample = np.concatenate((acc_time, gyr_time, acc_freq, gyr_freq), axis = 0)
    
    # Normalizing with zscore with mean and std of whole dataset
    features_mean = np.concatenate((features_mean_time, features_mean_freq))
    features_std = np.concatenate((features_std_time, features_std_freq))

    sample_normalized = (sample - features_mean)/features_std
    

    #Predict the response for test dataset
    sample_pred = model_sigmoid.predict(sample_normalized.reshape(1,-1))
    sample_pred2 = model_linear.predict(sample_normalized.reshape(1,-1))
    # print("Sgmoid : " + str(sample_pred))
    # print("Linear" + str(sample_pred2))
    
    
    # Model Accuracy: how often is the classifier correct?
    #print("Prediction:", sample_pred)
    
    return sample_pred



    



