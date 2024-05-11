
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
import os

LEARNING_RATE = 0.03


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #H1 = W1^t*X+W1_0
        #H2 = W2^t*relu(H1)+W2_0
        #Y_hat = W3^t*relu(H2)+W3_0
        #final prediction = softmax(Y_hat)
        self.H1 = nn.Linear(in_features=784, out_features=100)#W1(784*100),W1_0(100*1)
        self.H2 = nn.Linear(in_features=100, out_features=100)#W2(100*100),W2_0(100*1)
        self.Y_hat = nn.Linear(in_features=100, out_features=10)#W3(100*10),W3_0(10*1)
        
        # Initialize weights with a normal distribution, biases with zeros
        nn.init.normal_(self.H1.weight, std=0.1)
        nn.init.constant_(self.H1.bias, 0)
        nn.init.normal_(self.H2.weight, std=0.1)
        nn.init.constant_(self.H2.bias, 0)
        nn.init.normal_(self.Y_hat.weight, std=0.1)
        nn.init.constant_(self.Y_hat.bias, 0)

    def forward(self, x):
        x = F.relu(self.H1(x))
        x = F.relu(self.H2(x))
        x = F.softmax(self.Y_hat(x), dim=1)
        return x
    def get_H2(self, x):
        x = F.relu(self.H1(x))
        x = F.relu(self.H2(x))
        return x
    def get_H1(self, x):
        x = F.relu(self.H1(x))
        return x
    
#initialize training model and test model
model = MyModel()
test_model = MyModel()

class My_dataset(Dataset):
    """
    Dataset Class for any dataset.
    This is a python class object, it inherits functions from 
    the pytorch Dataset object.
    For anyone unfamiliar with the python class object, see 
    https://www.w3schools.com/python/python_classes.asp
    or a more complicated but more detailed tutorial
    https://docs.python.org/3/tutorial/classes.html
    For anyone familiar with python class, but unfamiliar with pytorch
    Dataset object, see 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(self, data_dir, anno_csv) -> object:
        self.anno_data = pd.read_csv(anno_csv)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.anno_data)

    def __getitem__(self, idx):
        data_name = self.anno_data.iloc[idx, 0]
        data_location = self.data_dir + data_name
        data = np.float32(np.load(data_location))
        # This is for one-hot encoding of the output label
        gt_y = np.float32(np.zeros(10))
        index = self.anno_data.iloc[idx, 1]
        gt_y[index] = 1
        return data, gt_y

#yield randomly selected mini batch from dataset with mini batch size
def get_random_mini_batch(dataset, mini_batch_size):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    for start_idx in range(0, dataset_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, dataset_size)
        mini_batch_indices = indices[start_idx:end_idx]
        
        mini_batch_data = []
        mini_batch_labels = []
        
        for idx in mini_batch_indices:
            data, label = dataset[idx]
            mini_batch_data.append(data)
            mini_batch_labels.append(label)
        
        mini_batch_data = np.array(mini_batch_data)
        mini_batch_labels = np.array(mini_batch_labels)
        
        yield torch.tensor(mini_batch_data), torch.tensor(mini_batch_labels)



def PA2_train():
    # Specifying the training directory and label files
    train_dir = './'
    train_anno_file = './data_prog2Spring24/labels/train_anno.csv'
    test_dir = './'
    test_anno_file = './data_prog2Spring24/labels/test_anno.csv'

    # Specifying the device to GPU/CPU. Here, GPU means 'cuda' and CPU means 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read the data and labels from the training data
    MNIST_training_dataset = My_dataset(data_dir=train_dir, anno_csv=train_anno_file)
    MNIST_testing_dataset = My_dataset(data_dir=test_dir, anno_csv=test_anno_file)
    
    #Get test data
    test_x = []
    test_y = []
    for i in range(4000):
        data, label = MNIST_testing_dataset[i]
        test_x.append(data)
        test_y.append(label)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    #Get training data
    train_x = []
    train_y = []
    for i in range(50000):
        data, label = MNIST_training_dataset[i]
        train_x.append(data)
        train_y.append(label)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)

    #You can set up your own maximum epoch. You may need  5 or 10 epochs to have a correct model.
    my_max_epoch = 10
    epochs = np.arange(0, my_max_epoch)
    loss_data=[]
    accuracy_data=[]
    class_data = [[],[],[],[],[],[],[],[],[],[]]
    test_class_data = [[],[],[],[],[],[],[],[],[],[]]
    train_class_data = [[],[],[],[],[],[],[],[],[],[]]
    
    total_batch = 1000
    it =0
    for epoch in epochs:
        
            
            
        #Randomly split your training data into mini-batches where each mini-batch has 50 samples
        #Since we have 50000 training samples, and each batch has 50 samples,
        #the total number of batch will be 1000
        # YOU ARE NOT ALLOWED TO USE DATALOADER CLASS FOR RANDOM BATCH SELECTION
        for train_features, train_labels in get_random_mini_batch(MNIST_training_dataset, 50):
            
            it= it + 1
            print(it)
            
            #caculate loss and inaccuracy of minibatch
            y_hat = model.forward(train_features)

            digit_appearances = [0 for _ in range(10)]
            digit_errors = [0 for _ in range(10)]
            train_loss = 0
            for m in range(50):

                k = k = np.where(train_labels[m].numpy() == 1)[0].item()
                train_loss += torch.log(y_hat[m, k]).detach().numpy()
                digit_appearances[k] += 1
                classification = torch.argmax(y_hat[m]).item()
                if classification != k:
                    digit_errors[k] += 1

            train_loss = -train_loss
            train_inaccuracy = [digit_errors[i] / digit_appearances[i] if digit_appearances[i] != 0 else 0 for i in range(10)]
            
            for i in range(10):
                class_data[i].append(train_inaccuracy[i])
            avg_train_inaccuracy = sum(train_inaccuracy) / 10
            print(avg_train_inaccuracy)
            loss_data.append(train_loss)
            accuracy_data.append(avg_train_inaccuracy)
            
            

            
            #initialize chain rule elements
            dSigZ3_by_dZ3 = torch.zeros(50, 10, 10)
            dZ3_by_dW = torch.zeros(50, 100, 10, 10)
            dPhiZ2_by_dZ = torch.zeros(50, 100, 100)
            dZ2_by_dW = torch.zeros(50, 100, 100, 100)
            dPhiZ1_by_dZ = torch.zeros(50, 100, 100)
            dZ1_by_dW = torch.zeros(50, 784, 100, 100)
            
            #calculate the gradient of H2, W3 and W3_0
            Y_out_g = -1 * (train_labels / y_hat).type(torch.float32)
            H2 = model.get_H2(train_features)
            H1 = model.get_H1(train_features)
            for m in range(50):
                SigZ3 = y_hat[m] 
                
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            dSigZ3_by_dZ3[m, i, j] = SigZ3[i] * (1 - SigZ3[i])
                            dZ3_by_dW[m, :, i, j] = H2[m, :]
                        else:
                            dSigZ3_by_dZ3[m, i, j] = -SigZ3[j] * SigZ3[i]
            
           
            W3_adj = model.Y_hat.weight.repeat(50, 1).view(50, 100, 10)
            Y_out_g = Y_out_g.reshape(50, 10, 1)
            test = torch.matmul(W3_adj, dSigZ3_by_dZ3)
            H2_g = torch.matmul(test, Y_out_g)

            
            # dtype check
            if dSigZ3_by_dZ3.dtype != Y_out_g.dtype:
                Y_out_g = Y_out_g.type(dSigZ3_by_dZ3.dtype)
            
            
            W3_0_g = torch.matmul(dSigZ3_by_dZ3, Y_out_g)
            
            W3_g = torch.zeros(50, 100, 10)
            chain = torch.matmul(dSigZ3_by_dZ3, Y_out_g)
            
            for m in range(50):
                W3_g[m] = torch.matmul(dZ3_by_dW[m], chain[m]).view(100, 10)


            
            
            #calculate the gradient of H1, W2 and W2_0


            for m in range(50):
                reluZ2 = H2[m]  
            
                # 'diagonals'
                for i in range(100):
                    if reluZ2[i] > 0:
                        dPhiZ2_by_dZ[m, i, i] = 1
                    dZ2_by_dW[m, :, i, i] = H1[m, :]
            
            W2_adj = model.H2.weight.repeat(50, 1).view(50, 100, 100)
            test = torch.matmul(W2_adj, dPhiZ2_by_dZ)
            H1_g = torch.matmul(torch.matmul(W2_adj, dPhiZ2_by_dZ), H2_g)
            W2_0_g = torch.matmul(dPhiZ2_by_dZ, H2_g)
            
            
            W2_g = torch.zeros(50, 100, 100)
            chain = torch.matmul(dPhiZ2_by_dZ, H2_g)
            for m in range(50):
                W2_g[m] = torch.matmul(dZ2_by_dW[m], chain[m]).view(100, 100)
                
            
            
            #calculate the gradient of W1 and W1_0

            
            for m in range(50):
                reluZ1 = H1[m]  
                
                #'diagonals'
                for i in range(100):
                    if reluZ1[i] > 0:
                        dPhiZ1_by_dZ[m, i, i] = 1
                    dZ1_by_dW[m, :, i, i] = train_features[m, :]  
            
           
            W1_0_g = torch.matmul(dPhiZ1_by_dZ, H1_g)
            
            
            W1_g = torch.zeros(50, 784, 100)
            chain = torch.matmul(dPhiZ1_by_dZ, H1_g)
            for m in range(50):
                W1_g[m] = torch.matmul(dZ1_by_dW[m], chain[m]).view(784, 100)
            
            
            #get average of 50 samples
            W1_g_avg = W1_g.float().mean(0)
            W1_g_avg = torch.transpose(W1_g_avg, 0, 1)

            W1_0_g_avg = W1_0_g.float().mean(0)
            W1_0_g_avg = W1_0_g_avg.reshape((100))

            W2_g_avg = W2_g.float().mean(0)
            W2_g_avg = torch.transpose(W2_g_avg, 0, 1)

            W2_0_g_avg = W2_0_g.float().mean(0)
            W2_0_g_avg = W2_0_g_avg.reshape((100))

            W3_g_avg = W3_g.float().mean(0)
            W3_g_avg = torch.transpose(W3_g_avg, 0, 1)

            W3_0_g_avg = W3_0_g.float().mean(0)
            W3_0_g_avg = W3_0_g_avg.reshape((10))

            

            #gradient desent and update weight
            new_W3 = model.Y_hat.weight - LEARNING_RATE * W3_g_avg
            new_W3_0 = model.Y_hat.bias - LEARNING_RATE * W3_0_g_avg
            
            new_W2 = model.H2.weight - LEARNING_RATE * W2_g_avg
            new_W2_0 = model.H2.bias - LEARNING_RATE * W2_0_g_avg
            
            new_W1 = model.H1.weight - LEARNING_RATE * W1_g_avg
            new_W1_0 = model.H1.bias - LEARNING_RATE * W1_0_g_avg
            
            
            with torch.no_grad():
                model.Y_hat.weight.copy_(new_W3)
                model.Y_hat.bias.copy_(new_W3_0)
                model.H2.weight.copy_(new_W2)
                model.H2.bias.copy_(new_W2_0)
                model.H1.weight.copy_(new_W1)
                model.H1.bias.copy_(new_W1_0)
         
                
         
            
        #get test inaccuracy over epoch 
        y_test_hat = model.forward(test_x)
        
        test_digit_appearances = [0 for _ in range(10)]
        test_digit_errors = [0 for _ in range(10)]
        
        for m in range(4000):
    
            k = k = np.where(test_y[m].numpy() == 1)[0].item()
            
            test_digit_appearances[k] += 1
            classification = torch.argmax(y_test_hat[m]).item()
            if classification != k:
                test_digit_errors[k] += 1
    
        test_inaccuracy = [test_digit_errors[i] / test_digit_appearances[i] if test_digit_appearances[i] != 0 else 0 for i in range(10)]
    
        for i in range(10):
            test_class_data[i].append(test_inaccuracy[i])
            
            
            
        #get train inaccuracy over epoch  
        y_train_hat = model.forward(train_x)
        
        train_digit_appearances = [0 for _ in range(10)]
        train_digit_errors = [0 for _ in range(10)]
        
        for m in range(50000):
    
            
            k = k = np.where(train_y[m].numpy() == 1)[0].item()
            
            train_digit_appearances[k] += 1
            classification = torch.argmax(y_train_hat[m]).item()
            if classification != k:
                train_digit_errors[k] += 1
    
        train_ep_inaccuracy = [train_digit_errors[i] / train_digit_appearances[i] if train_digit_appearances[i] != 0 else 0 for i in range(10)]
    
        for i in range(10):
            train_class_data[i].append(train_ep_inaccuracy[i])
       

    
    
    # Plot the training loss vs iteration
    iteration = np.arange(0, 10000)
    plt.plot(iteration, loss_data)
    plt.xlabel('iteration')

    plt.ylabel('loss')

    plt.title('training loss')

    plt.show()
    

    #Plot the training and testing errors vs iteration
    iter_epoch = np.arange(1, 11)
    
    for i in range(10):
        plt.plot(iter_epoch, train_class_data[i])
        plt.xlabel('epoch')

        plt.ylabel('inaccuracy')

        plt.title('training inaccuracy class'+str(i+1))

        plt.show()
    
    for i in range(10):
        plt.plot(iter_epoch, test_class_data[i])
        plt.xlabel('epoch')

        plt.ylabel('inaccuracy')

        plt.title('test inaccuracy class'+str(i+1))

        plt.show()
    
    
    # Visualize the final weight matrix
    W_f = torch.transpose(model.Y_hat.weight, 0, 1)
    #W = model.Y_hat.weight
    for k in range(10):
        Wk = W_f[:, k:k+1]
        Wk = Wk[:100]
        print(Wk.size())
        img = Wk.detach().numpy().reshape(10, 10)

    # Plotting
        plt.imshow(img)
        plt.colorbar()
        plt.show()
    # Save the final weight matrix
    theta = [model.H1.weight,model.H1.bias,model.H2.weight,model.H2.bias,model.Y_hat.weight,model.Y_hat.bias]
    f = open("nn_parameters.txt", "wb")
    pickle.dump(theta, f, protocol=2)
    f.close()
def PA2_test():
    # Specifying the training directory and label files
    test_dir = './'
    test_anno_file = './data_prog2Spring24/labels/test_anno.csv'
    feature_length = 784
    # Specifying the device to GPU/CPU. Here, GPU means 'cuda' and CPU means 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load the Weight Matrix that has been saved after training
    filehandler = open("nn_parameters.txt", "rb")
    W = pickle.load(filehandler)
    with torch.no_grad():
        test_model.Y_hat.weight.copy_(W[4])
        test_model.Y_hat.bias.copy_(W[5])
        test_model.H2.weight.copy_(W[2])
        test_model.H2.bias.copy_(W[3])
        test_model.H1.weight.copy_(W[0])
        test_model.H1.bias.copy_(W[1])
    

    # Read the data and labels from the testing data
    MNIST_testing_dataset = My_dataset(data_dir=test_dir, anno_csv=test_anno_file)
    test_x = []
    test_y = []
    for i in range(4000):
        data, label = MNIST_testing_dataset[i]
        test_x.append(data)
        test_y.append(label)
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)
    # Predict Y using X and updated W.
    y_test_hat = test_model.forward(test_x)
    
    test_digit_appearances = [0 for _ in range(10)]
    test_digit_errors = [0 for _ in range(10)]
    
    for m in range(4000):
    
        k = k = np.where(test_y[m].numpy() == 1)[0].item()
        
        test_digit_appearances[k] += 1
        classification = torch.argmax(y_test_hat[m]).item()
        if classification != k:
            test_digit_errors[k] += 1
            
    # Calculate accuracy,
    test_inaccuracy = [test_digit_errors[i] / test_digit_appearances[i] if test_digit_appearances[i] != 0 else 0 for i in range(10)]
    print(test_inaccuracy)
    avg_test_inaccuracy = sum(test_inaccuracy) / 10
    print(avg_test_inaccuracy)
    
        
if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    #PA2_train()
    PA2_test()
