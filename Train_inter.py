import numpy as np
import matplotlib.pyplot as plt
import torch, math
import LoadData, T_Model, utils
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error
import argparse
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intelligent Knee sleeve')
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--seq_length', type=int, default=220, help="sequence length for sliding window")
    parser.add_argument('--max_epoch', type=int, default=2000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--lr', type=float, default=1e-6, help="learning rate")
    parser.add_argument('--input_size', type=int, default=36, help="input data dimension")
    parser.add_argument('--hidden_size', type=int, default=10, help="hidden size of LSTM")
    parser.add_argument('--num_layers', type=int, default=2, help="number of LSTM layers")
    parser.add_argument('--num_classes', type=int, default=32, help="output data dimension")
    parser.add_argument('--type', type=str, default='time', help="plot and compare quaternion outputs")
    parser.add_argument('--patience', type=int, default=5, help="plot and compare quaternion outputs")
    parser.add_argument('--train_type', type=str, default='all_seen', choices=['all_seen', 'unseen_tasks','unseen_date','unseen_people'])
    parser.add_argument('--unseen_type', type=str, default='bendsquat', choices=['bendsquat', 'hamstring','legraise','legraise_90'])
    parser.add_argument('--Test_day', type=int, default=7)
    parser.add_argument('--Test_pid', type=int, default=1)

    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # define the training and testing files
    folder = './dataset/'
    if args.train_type == 'all_seen':
        files = folder+'summary.csv'
        files = folder+utils.get_filename(files)
        test_files = random.sample(list(files), int(0.1*len(files)))
        files = list(set(files) - set(test_files))
        selected_pose = "all_seen"

    if args.train_type == 'unseen_tasks':
        files = folder+'unseen_task.csv'
        files = folder+utils.get_filename(files)
        if args.unseen_type == 'bendsquat':
            test_files=folder+'Bendsquat.csv'
            test_files = folder+utils.get_filename(test_files)
            print("unseen tasks - bendsquat")    

        if args.unseen_type == 'hamstring':
            test_files=folder+'hamstring.csv'
            test_files = folder+utils.get_filename(test_files)    
            print("unseen tasks - hamstring")   

        if args.unseen_type == 'legraise':
            test_files=folder+'LegRaise.csv'
            test_files = folder+utils.get_filename(test_files)  
            print("unseen tasks - legraise")     

        if args.unseen_type == 'legraise_90':
            test_files=folder+'LegRaise.csv'
            test_files = folder+utils.get_filename(test_files)   
            test_files = random.sample(list(test_files), int(0.9*len(test_files)))
            print("unseen tasks - legraise_90%")    

        files = list(set(files) - set(test_files))
        selected_pose = args.unseen_type

    if args.train_type == 'unseen_date':
        files = folder+'summary.csv'
        files = folder+utils.get_filename(files)
        test_files = files[files.str.contains('D'+str(args.Test_day))]
        files = list(set(files) - set(test_files))
        selected_pose = args.train_type

    if args.train_type == 'unseen_people':
        files = folder+'summary.csv'
        files = folder+utils.get_filename(files)
        test_files = files[files.str.contains('P'+str(args.Test_pid))]
        files = list(set(files) - set(test_files))
        selected_pose = args.train_type

    # split the validation and training files 
    val_files = random.sample(list(files), int(0.1*len(files)))
    files = list(set(files) - set(val_files))

    transform = MinMaxScaler(feature_range=(-1, 1))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("start working on: " + selected_pose)

    print(len(files))
    print(len(test_files))

    print(files)
    print(test_files)


    train_dataset = LoadData.KneeDataset(filenames=files, batch_size=args.batch_size, seq_length=args.seq_length,
                                        transform=transform)
    val_dataset = LoadData.KneeDataset(filenames=val_files, batch_size=args.batch_size, seq_length=args.seq_length,
                                        transform=transform)
    test_dataset = LoadData.KneeDataset(filenames=test_files, batch_size=args.batch_size, seq_length=args.seq_length,
                                        transform=transform, test=1)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = T_Model.LSTM(args.num_classes, args.input_size, args.hidden_size, args.num_layers, args.seq_length, dev)
    # model =T_Model.CNNNet(num_classes=params.num_classes)

    model.to(dev)
    
    # criterion = torch.nn.MSELoss()  # mean-squared error for regression
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=32)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    val_loss = []
    train_loss = []

    # Train the model
    for epoch in range(args.num_epochs):
        running_loss_train = 0.0
        running_loss_test = 0.0
        correct_train = 0.0
        correct_test = 0.0
        num_labels_train = 0.0
        num_labels_test = 0.0
        running_loss_val = 0.0

        for data, labels in train_dataloader:
            model.train()
            # data = data.reshape(-1, params.seq_length, params.input_size).float().to(params.dev)
            data, labels = data.float().to(dev), labels.float().to(dev)
            outputs = model(data)
            optimizer.zero_grad()

            # print(outputs.device)
            loss = criterion(outputs, labels) # obtain the loss function
            loss.backward()
            optimizer.step()
            # scheduler.step()
            running_loss_train = running_loss_train + loss.item()

        # validate the model:
        with torch.no_grad():
            for data, labels in val_dataloader:
                model.eval()
                data, labels = data.float().to(dev), torch.squeeze(labels).float().to(dev)
                outputs = model(data)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()

        # Early stopping check
        early_stopping((running_loss_val), model)
        if early_stopping.early_stop:
            print("Early stopping at epoch", epoch)
            break
        
        val_loss.append(running_loss_val / len(val_dataloader))
        train_loss.append(running_loss_train / len(train_dataloader))

        test_output = []
        labels_output=[]
        for i, (data, labels) in enumerate(test_dataloader):
            model.eval()
            data, labels = data.float().to(dev), labels.float().to(dev)  
            outputs = model(data)
            # np.concatenate((outputs, test_output), axis=0)
            loss = criterion(outputs, labels)
            running_loss_test = running_loss_test + loss.item()
            # test_output.extend(params.transform.inverse_transform(outputs.cpu().detach().numpy()))  
            # labels_output.extend(params.transform.inverse_transform(labels.cpu().detach().numpy())) 
            test_output.extend(outputs.cpu().detach().numpy())  
            labels_output.extend(labels.cpu().detach().numpy())
        # if epoch %  == 0:
        print("Epoch:{}, Train_loss: {:.4f}, Val_loss: {:.4f}". \
            format(epoch, running_loss_train / len(train_dataloader),
                    running_loss_test / len(test_dataloader)))


    plt.plot(train_loss), plt.xlabel("training loss"), plt.savefig(r'./dataset//'+selected_pose+'training_loss.png'), plt.show(),plt.close()
    plt.plot(val_loss), plt.xlabel("validation loss"), plt.savefig(r'./dataset//'+selected_pose+'val_loss.png'), plt.show(),plt.close()


    test_output = np.asarray(test_output)
    test_dataset.subject_dict.to_csv(r'./dataset/'+selected_pose+'sub_info.csv',index=True)
    # _, filename=split(params.files[-1][0])
    np.savetxt(r'./dataset/'+selected_pose+'prediction.csv', test_output, delimiter=',')
    labels_output = np.asarray(labels_output)
    np.savetxt(r'./dataset/'+selected_pose+'ground_.csv', labels_output, delimiter=',')
    RMSE = math.sqrt(mean_squared_error(test_output, labels_output))
    print("Root Mean Square: ", RMSE)
    r2_score = 1 - mean_squared_error(test_output, labels_output) / np.var(test_output)
    print("R Squared Error: ", r2_score)
    plt.plot(test_output), plt.title('predicted output'), plt.show()
    plt.savefig(r'dataset//'+selected_pose+'predicted.png', bbox_inches='tight'), plt.clf()
    plt.plot(labels_output),  plt.title('actual output')
    plt.savefig(r'dataset//'+selected_pose+'labels_output.png', bbox_inches='tight'), plt.clf()
    # utils.plot(labels_output, test_output, params.num_classes, params.output_title[0], params.type)
