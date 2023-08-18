import csv, math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import savefig, xlim, figure, ylim, legend, boxplot, setp, axes
import seaborn as sns
from scipy.spatial.transform import Rotation

# set the parameters
plt.rcParams['font.size'] = 15
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 6

def plot_compare(time_stamp,pre,grd):

    sides = ["left","right"]
    start_columns=np.array([0,16])

    quat_label = ['pre_x','pre_y','pre_z','pre_w']
    grd_label = ['grd_x','grd_y','grd_z','grd_w']
    grd_label = ['x','y','z','w']
    fig, axs = plt.subplots(4, 2, figsize=(8, 8), sharex=True)
    markersize=6
    markevery = 35
    for i in np.arange(2):
        side = sides[i]
        start_column = start_columns[i]
        # for kk in np.arange(4):
        axs[0,i].plot(time_stamp,pre[:,start_column:4+start_column], marker='o',markersize=markersize,markevery=markevery)
        axs[0,i].plot(time_stamp,grd[:,start_column:4+start_column], marker='x',markersize=markersize,markevery=markevery)
        axs[0,i].set_title('Hip angle ('+ side + ')')
        
        axs[1,i].plot(time_stamp,pre[:,4+start_column:8+start_column],marker='o',markersize=markersize,markevery=markevery)
        axs[1,i].plot(time_stamp,grd[:,4+start_column:8+start_column],marker='x',markersize=markersize,markevery=markevery),
        axs[1,i].set_title('Knee angle ('+ side + ')')

        axs[2,i].plot(time_stamp,pre[:,8+start_column:12+start_column],marker='o',markersize=markersize,markevery=markevery)
        axs[2,i].plot(time_stamp,grd[:,8+start_column:12+start_column],marker='x',markersize=markersize,markevery=markevery),
        axs[2,i].set_title('Ankel angle ('+ side + ')')

        axs[3,i].plot(time_stamp, pre[:,12+start_column:16+start_column],marker='o',markersize=markersize,markevery=markevery,label=quat_label)
        axs[3,i].plot(time_stamp,grd[:,8+start_column:12+start_column],marker='x',markersize=markersize,markevery=markevery,label=grd_label),
        axs[3,i].set_title('Toe angle ('+ side + ')')
    # plt.xlabel('Time (s) \n Compare of ground truth and predictions')
    fig.text(0.5, -0.01, 'Time (s) \n Compare of ground truth and predictions', ha='center')
    xlabel = fig.axes[-1].xaxis.label
    xlabel.set_size(20)
    xlabel.set_weight('bold')
    axs[3,1].legend()
    fig.tight_layout() 
    plt.show()

def cal_RMSE(pred,grd):
    # MSE / MAE error in the unit of degrees
    rmse=np.zeros((1,8))
    diff_array = np.zeros(pred.shape)
    diff_array = np.abs(pred)-np.abs(grd)
    rmse_xyz = np.zeros((3,8))
  
    for i in np.arange(8):
        sq_diff_x = [d ** 2 for d in diff_array[:,i*3]]
        sq_diff_y = [d ** 2 for d in diff_array[:,i*3+1]]
        sq_diff_z = [d ** 2 for d in diff_array[:,i*3+2]]
        
        mean_sq_diff = (sum(sq_diff_x) + sum(sq_diff_y) + sum(sq_diff_z)) / (3 * len(grd))
        rmse[0,i] = math.sqrt(mean_sq_diff)
        rmse_xyz[0,i] = math.sqrt(sum(sq_diff_x)/len(grd)) 
        rmse_xyz[1,i] = math.sqrt(sum(sq_diff_y)/len(grd)) 
        rmse_xyz[2,i] = math.sqrt(sum(sq_diff_z)/len(grd)) 
    
    return rmse, rmse_xyz

def quat2euler(quat):
    rot = np.zeros((quat.shape))
    rot_euler = np.zeros((quat.shape[0],int(quat.shape[1]/4*3)))
    for i in np.arange(8):
        rot = Rotation.from_quat(quat[:,i*4:i*4+4])
        rot_euler[:,i*3:i*3+3] = rot.as_euler('xyz', degrees=True)
    return rot_euler

def quat_distance(pred,grd):
    # 8 joints, each joint has a distance
    distance = np.zeros((pred.shape[0], 8))
    for i in np.arange(8):
        distance[:,i] = 1 - np.power((pred[:,i*4]*grd[:,i*4]) + (pred[:,i*4+1]*grd[:,i*4+1]) + 
                                (pred[:,i*4+2]*grd[:,i*4+2]) + (pred[:,i*4+3]*grd[:,i*4+3]),2)
    return distance

def quat_boxplot_seperate(data,string,colors):
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,notch=True)
    for patch in bp['boxes']:
        patch.set_facecolor(colors)

    for elemment in bp:
        for lines in bp[elemment]:
            lines.set_linewidth(2)
        
    ax.set_xticklabels(['LHip','LKnee','LAnkel','LToe','RHip','RKnee','RAnkel','RToe'])
    ax.set_xlabel(string)
    # ax.set_title()
    fig.tight_layout() 
    plt.xticks(rotation=15)
    plt.show()   

def Unseen_tasks(folder,colors):
    unseen_tasks = ["Unseen_BendSquat","Unseen_Hamstring","Unseen_LegRaise","All_Seen"]  
    folder = folder+"/UnseenTasks/"
    rmse = np.zeros((8,len(unseen_tasks)))
    rmse_xyz = np.zeros((len(unseen_tasks)*3,8))
    pred_dgre_avg = []
    grd_dgre_avg = []
    for index, unseen_task in enumerate(unseen_tasks):
        prediction = pd.read_csv(folder+unseen_task+'_prediction.csv',header=None).values
        grd = pd.read_csv(folder+unseen_task+'_ground_.csv',header=None).values
        quat_distances = quat_distance(prediction,grd)
        quat_boxplot_seperate(quat_distances,unseen_task[7:],colors[index])
        pred_dgree=quat2euler(prediction)
        grd_dgree=quat2euler(grd)
        rmse[:,index], rmse_xyz[index*3:index*3+3,:] = cal_RMSE(pred_dgree,grd_dgree)
        pred_dgre_avg.extend(pred_dgree)
        grd_dgre_avg.extend(grd_dgree)
    rmse_avg,rmse_xyz_all = cal_RMSE(np.array(pred_dgre_avg),np.array(grd_dgre_avg))
    print("finish")

def all_seen(folder,colors_3):
    file = "All_seen"
    pred = pd.read_csv(folder+"/UnseenTasks/"+file+"_prediction.csv",header=None).values
    grd = pd.read_csv(folder+"/UnseenTasks/"+file+"_ground_.csv",header=None).values
    sub_info=pd.read_csv(folder+"/UnseenTasks/"+file+"_sub_info.csv", index_col=0)
    pred_dgree=quat2euler(pred)
    grd_dgree=quat2euler(grd)
    quat_distances_suqat =[]
    quat_distances_hamstring = []
    quat_distances_legraise = []
    dgr_squat_pred=[]
    dgr_squat_grd = []
    dgr_hamstring_pred=[]
    dgr_hamstring_grd=[]
    dgr_legraise_pred=[]
    dgr_legraise_grd = []
    rmse_per,rmse_xyz = cal_RMSE(pred_dgree,grd_dgree)
    for index in sub_info.index:
        start_step = sub_info['start_steps'][index]
        end_step = sub_info['end_steps'][index]
        quat_error_per = quat_distance(pred[start_step:end_step,:],grd[start_step:end_step,:])
        time_stamp = np.linspace(0,(end_step-start_step)/f,end_step-start_step)
        if 'Squat' in sub_info['filename'][index]:
            quat_distances_suqat.extend(quat_error_per)
            dgr_squat_pred.extend(pred_dgree[start_step:end_step,:])
            dgr_squat_grd.extend(grd_dgree[start_step:end_step,:])
            # plot_compare(time_stamp, pred[start_step:end_step,:],grd[start_step:end_step,:])
        elif 'Hamstring' in sub_info['filename'][index]:
            quat_distances_hamstring.extend(quat_error_per)
            dgr_hamstring_pred.extend(pred_dgree[start_step:end_step,:])
            dgr_hamstring_grd.extend(grd_dgree[start_step:end_step,:])
        elif 'LegRaise' in sub_info['filename'][index]:
            quat_distances_legraise.extend(quat_error_per)
            dgr_legraise_pred.extend(pred_dgree[start_step:end_step,:])
            dgr_legraise_grd.extend(grd_dgree[start_step:end_step,:])
            # plot_compare(time_stamp, pred[start_step:end_step,:],grd[start_step:end_step,:])
        else:
            print(sub_info['filename'][index] + " unrecognized")      

    quat_boxplot_seperate(quat_distances_suqat,"Squat",colors_3[0])  
    quat_boxplot_seperate(quat_distances_hamstring,"Hamstring",colors_3[1])
    quat_boxplot_seperate(quat_distances_legraise,"LegRaise",colors_3[2])
    rmse_per_squat,rmse_xyz_squat = cal_RMSE(np.array(dgr_squat_pred),np.array(dgr_squat_grd))
    rmse_per_hamstring,rmse_xyz_hamstring = cal_RMSE(np.array(dgr_hamstring_pred),np.array(dgr_hamstring_grd))
    rmse_per_legraise,rmse_xyz_legraise = cal_RMSE(np.array(dgr_legraise_pred),np.array(dgr_legraise_grd))

if __name__ == '__main__':
    folder = "./Results_visual/"
    f = 20
    colors_3 = ["lightblue", "#84ba09", "pink","#FDDD8C"]
    colors_3 = ["#ECA8A9","#74AED4","#D3E2B7","#CFAFD4","#F7C97E"]
    Unseen_tasks(folder,colors_3)
    all_seen(folder,colors_3)
    


    