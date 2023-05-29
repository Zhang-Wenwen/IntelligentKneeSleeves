import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import interpolate
import sys
from scipy.signal import medfilt
from scipy import signal
import math

def sliding_windows(data, labels, seq_length):
    x = []
    y = []

    for i in range(0, len(data) - seq_length - 1, 1):
        _x = data[i:(i + seq_length), :] 
        _y = labels[(i + seq_length)]  # only look at the last output
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

def inter_x(data):
    x = data[:,0]
    y=data[:,1:]
    even_x = np.linspace(x[0], x[-1], num=int((x[-1]-x[0])*20))

    # f = [interpolate.interp1d(x, y[i], kind='linear') for i in range(y.shape[0])] 
    # yinterp=np.array([f[i](even_x) for i in range(y.shape[0])]) 

    # # 20 is the sampling rate. Interpolate the signal to make time evenly increasing
    yinterp = signal.resample(y, len(even_x))
    return np.concatenate((even_x.reshape(-1,1), yinterp),axis=1)

def rm_abnormal(data):
    # remove open circuit pin, which will be very large

    indices = np.where(abs(data[:,1:15]) > 0.8)
    columns = set(indices[1]) 
    # check abnormal values for each column
    for column in columns:
        indices = np.where(abs(data[:,column+1]) > 0.8)    
        rows = set(indices[0])
        
        if len(rows)>0.1*data.shape[0]:
            data=avg_pin(data,column+1)
        else:
            data = np.delete(data, list(rows), axis=0)

    
    return data

def rm_sparkes(data):
    diff_pin=np.diff(data[:,1:15],axis=0)

    # calculate the mean and standard deviation of the array
    mean = np.mean(diff_pin,axis=0)
    std_dev = np.std(diff_pin,axis=0)

    # define a threshold for abnormal values (e.g., 3 standard deviations from the mean)
    threshold_v1 = mean + 10 * std_dev

    # abnormal_threshold means abnormal data as well 
    abnormal_thresholds = np.where(threshold_v1 > 0.4)
    for abnormal_threshold in abnormal_thresholds[0]:
        data=avg_pin(data,abnormal_threshold+1)

    for i in range(diff_pin.shape[1]):
        indices_v1 = np.where(np.abs(diff_pin[:,i]) > threshold_v1[i])
    
        # return directly if no sparke detected
        if len(indices_v1[0])==0:
            continue  
        
        column=(i+1)

        # if too many sparkes, replace the pins value with nearby rows
        if len(indices_v1[0])>0.1*data.shape[0]:
            # avg the values of the previous pin and afterward pin
            data=avg_pin(data,column)
            continue  
        
        data[:,column] = med_pass(data[:,column])

        # calculate the mean and standard deviation of the array
        mean = np.mean(data[:,column])
        std_dev = np.std(data[:,column])

        # define a threshold for abnormal values (e.g., 3 standard deviations from the mean)
        threshold = mean + 3 * std_dev

        indices = np.where(np.abs(data[:,column]) > threshold)

        if len(indices[0])>0.1*data.shape[0]:
            # check_pin_dim(column)
            data=avg_pin(data,column)           
            continue  

        # rows = set(indices[0])
        # # remove the rows from the array
        # data = np.delete(data, list(rows), axis=0)

        # print(len(rows))
        data = abnormal_data(data, column)

        data_temp = data[:,column]
        data_temp[indices[0]] = np.nan

        for i in np.arange(len(indices[0])):
            abnormal_row=indices[0][i]
            window_size=6
            # calculate the average of the nearby points
            row_start = max(0, abnormal_row - window_size)
            row_end = min(data.shape[0], abnormal_row + window_size + 1)

            nearby_points = data_temp[row_start:row_end]
            nearby_points = nearby_points[~np.isnan(nearby_points)]
            # nearby_points = nearby_points[nearby_points != data[abnormal_row-2:abnormal_row+3, abnormal_col-2:abnormal_col+3]]
    
            avg = np.mean(nearby_points)

            # replace the abnormal value with the calculated average
            data[abnormal_row, column] = avg

    return data

def abnormal_data(data,column):
    data_temp = np.delete(data, column, axis=1)
    # data_temp[:,column]=[]
    threshold = np.max(data_temp[:,1:14])*1.5
    indices = np.where(np.abs(data[:,column]) > threshold)

    if len(indices[0])>0.1*data.shape[0]:
        # avg the values of the previous pin and afterward pin
        data=avg_pin(data,column)

    return data

def med_pass(data):
    window_size = 5
    data = medfilt(data, window_size)
    return data

def rm_short(data):
    # remove short pins, which is -1 in sensor values
    indices = np.where(data[:,1:15] == -1)
    if len(indices[1])==0:
        return data
    
    # get the rows that contain elements greater than the threshold
    column = set(indices[1])

    check_pin_dim(column)

    # avg the values of the previous pin and afterward pin
    column=(int(column.pop())+1)

    data = avg_pin(data,column)

    return data


def check_pin_dim(column):
    if len(column)>1:
        print("Error: more than 1 pin is depracted")
        # sys.exit("Exiting program...")

def avg_pin(data,column):
    if column<14 & column>1:
        data[:,column] = (data[:,column-1]+data[:,column+1])/2
    elif column >1:
        data[:,column] = data[:,column-1] 
    else:
        data[:,column] = data[:,column+1] 
    return data

def multiple_pin_failure(data, columns):
    # when multiple pins are not working
    columns = np.array(list(columns))
    for column in columns:
        data[:,column] = (data[:,column-1] + data[:,column-2]) /2
    print("multiple pin corrected")
    return data
        
        
def rm_nan(data):
    nan_indices = np.where(np.isnan(data) | np.isinf(data))
    column = set(nan_indices[1])

    if len(column)==0:
        return data


    if len(column)>1:
        data = multiple_pin_failure(data, column)
        return data
    
    # avg the values of the previous pin and afterward pin
    column=(int(column.pop()))

    if column<15:
        data[:,column] = (data[:,column-1]+data[:,column+1])/2
    else:
        data[column] = (data[:,column-1] + data[:,column-2]) /2
    return data

def length_check(data1,data2, data3):
    # make the label and knee file same length
    # the two sensor may not end at the same time

    data_len = np.min([data1.shape[0],data2.shape[0], data3.shape[0]])

    data1 = data1[0:data_len,:]

    data2 = data2[0:data_len,:]

    data3 = data3[0:data_len,:]

    return [np.concatenate((data1, data2), axis=1), data3]

def pre_process(data):
    # the process order is: first drop abnormal value and interpolate
    data=rm_nan(data)
    data=rm_abnormal(data)
    data=rm_abnormal(data)
    data=rm_short(data)
    data=rm_sparkes(data)
    data=rm_sparkes(data)
    data=inter_x(data)

    return data

def clean_for_norm(temp):
    # delete time and quaternion column for normalization
    temp = np.delete(temp, 0, axis=1)
    temp = np.delete(temp, 18, axis=1)
    q_temp_l = temp[:,14:18]
    q_temp_r = temp[:,32:36]

    temp = np.delete(temp, np.arange(32,36), axis=1)
    temp = np.delete(temp, np.arange(14,18), axis=1)

    return temp, q_temp_l, q_temp_r

def extra_info(filename,init_step,num_files,subject_dict):
    subject_dict=subject_dict.append({"filename": filename, "start_steps":init_step, "end_steps": num_files}, 
                                    ignore_index=True)
    return subject_dict

def rotate_quat(q):
    angle =-math.pi / 2
    axis = [0, 1, 0]
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    q_rot = [c, s*axis[0], s*axis[1], s*axis[2]]
    q_new = [
        q[0]*q_rot[0] - q[1]*q_rot[1] - q[2]*q_rot[2] - q[3]*q_rot[3],
        q[0]*q_rot[1] + q[1]*q_rot[0] + q[2]*q_rot[3] - q[3]*q_rot[2],
        q[0]*q_rot[2] - q[1]*q_rot[3] + q[2]*q_rot[0] + q[3]*q_rot[1],
        q[0]*q_rot[3] + q[1]*q_rot[2] - q[2]*q_rot[1] + q[3]*q_rot[0]
    ]
    return q_new

def rotate_quat_array(q):
    q_new = np.zeros(q.shape)
    for i in np.arange(q.shape[0]):
        q_new[i,:] =  rotate_quat(q[i,:])
    return q_new

class KneeDataset(Dataset):
    def __init__(self, filenames, batch_size, seq_length, transform,test=0):
        # `filenames` is a list of strings that contains all file names.
        # `batch_size` determines the number of files that we want to read in a chunk.
        self.filenames = filenames
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.transform = transform
        self.data = []
        self.labels = []
        self.subject_dict = pd.DataFrame({}) 
        init_step = 0 

        for file in self.filenames:
            # load left and right knee data
            labels_temp = pd.read_csv(open(file+'.csv', 'r')).values

            temp_l = pd.read_csv(open(file+'_l.csv', 'r')) 
            temp_l =  temp_l.loc[:, ['Time','P0','P1','P2','P3','P4','P5','P10','P11','P12','P13','P14','P15',
                                     'P22','P23','QW','QX','QY','QZ']].values

            try:
                temp_r = pd.read_csv(open(file+'_r.csv', 'r')) # Change this line to read any other type of file
                temp_r = temp_r.loc[:, ['Time','P0','P1','P2','P3','P4','P5','P10','P11','P12','P13','P14','P15',
                                     'P22','P23','QW','QX','QY','QZ']].values
            except:
                print(file+"  missing")
                continue
            
            temp_l=pre_process(temp_l)
            temp_r=pre_process(temp_r)

            temp,labels_temp = length_check(temp_l,temp_r,labels_temp)

            # temp = np.delete(temp, 0, axis=1)
            # temp = np.delete(temp, 18, axis=1)

            temp, q_temp_l, q_temp_r = clean_for_norm(temp)

            if 'LegRaise' in file:
                q_temp_l = rotate_quat_array(q_temp_l)
                q_temp_r = rotate_quat_array(q_temp_r)

            temp = self.transform.fit_transform(temp)
            # labels_temp = self.transform.fit_transform(labels_temp)

            # insert the quaternions
            temp = np.insert(temp, [14,14,14,14], q_temp_l, axis=1)
            temp = np.concatenate([temp, q_temp_r], axis=1)

            [temp, labels_temp] = sliding_windows(temp, labels_temp, self.seq_length)
            
            self.data.extend(temp)  # .reshape(-1,1,self.seq_length,60))
            self.labels.extend(labels_temp)  # .reshape(-1,1,self.seq_length,32))
            
            if test:
                self.subject_dict = extra_info(file,init_step,len(temp)+init_step,self.subject_dict)
                init_step += len(temp)

        self.data = np.asarray(
            self.data)  # .reshape(-1,1,self.seq_length,60)  # Because of Pytorch's channel first convention
        self.labels = np.asarray(self.labels)  # .reshape(-1,1,self.seq_length,32)

    def __len__(self):
        return self.data.shape[0]
        # return int(np.ceil(len(self.filenames) / float(self.batch_size)))  # Number of chunks.

    def __getitem__(self, index):  # index means index of the chunk.
        # In this method, we do all the preprocessing.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.

        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        label = self.labels[index, :] #.astype(float)
        data = self.data[index, :, :] #.astype(float)
        return data, label
