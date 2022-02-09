# NeuralNetworkTraining2.py

import torch 
import torch.nn as nn #  linear , loss function ... 
import torch.nn.functional as F  # All the functions that don't have any parameters like ReLu, tanh ... 
import torch.optim as optim  # all the optimization algotithms (stochastics gradient decent , Adam ... etc)
from torch.utils.data import DataLoader  # gives small data set management ( here the )
import torchvision.datasets as datasets # mnist datasetts 
import torchvision.transforms as transforms # perform transformation on the datasets 

# create the neural network 
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() # calls the initialization method of the parent class 
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# test 
model  = NN(784, 10) # 784 is the number of inout for each image 
print(model)
# generate a such example 
x = torch.randn(64, 784) # 64 is the number of examples that we gonna run simultaniously  (the batch size)
# run the model on this set
print(model(x).shape)


# set the device  

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)

# Hyper parameters 

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1 # number of epochs that we want to train 

##### prepare the train datasets 
# load the data 
train_dataset = datasets.MNIST(root='dataset/', train=True, transform = transforms.ToTensor(),download = True)
# need to create a training loader 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle= True)

##### prepare the test datasets 
test_dataset = datasets.MNIST(root='dataset/', train=False, transform = transforms.ToTensor(),download = True)
# need to create a training loader 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle= True)


# Initialise the network  

model = NN(input_size, num_classes).to(device) # imported to choose the device 

# loss and optimizer 

criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# Train the network



for epoch in range(num_epochs):   # for each epoch 
    for indx, (data, target) in enumerate(train_loader): 
        data = data.to(device = device)   # adapt the data to the device 
        target = target.to(device = device)

        # print(data.shape)
        #  format torch.Size([64, 1, 28, 28])  
        # 64 : number of examples ( of images )
        # 1 channel (white of black )  RGB channel = 3 
        # 28, 28 = shape of the images ( width, height)
        # so here we need to reshape the data
        # solution ==> 

        data = data.reshape(data.shape[0], -1) # -1 will flatten all the dimensions to a single dimensions 
        # pass the data over the network 
        scores = model(data)
        # comoute the loss 
        loss = criterian(scores, target) # as input the prediction of the neural network and the correct answer ( ground truth )
        # for ach batch we want to set the gradients to 0 so he donesn't store the previous back prop for the previous forward prop 
        optimizer.zero_grad()
        loss.backward()

        # gradient decent for adam step 
        optimizer.step()

        ### the training ends here ###


# at the end we will checl the accuracy of our neural network 

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # when we do the evaluation pythorch don't need to perform a computation of the gradiant 
        for x,y in loader: 
            x = x.to(device=device)
            y = y.to(device=device)

            x= x.reshape(x.shape[0], -1)
            
            scores = model(x) # here we will get a 64 * 10 tensor ( we only need the max for the 64 so we need to use dimension 0 ;) 
            _, prediction = scores.max(1)
            num_correct  += (prediction == y).sum() # the max will give the chosen prediction if prediction == y == > true / the sum will just convert from binary to float 
            num_samples += prediction.size(0)
        
        # print the accuracy 
        print(f'Got accuracy = {(float(num_correct) / float( num_samples)) * 100 } %' )
    model.train()
    

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)










