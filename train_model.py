#!/usr/bin/env python
# coding: utf-8

# In[1]:


from baseline_cnn import *
from baseline_cnn import BasicCNN
import torch



# Setup: initialize the hyperparameters/variables
num_epochs = 50           # Number of full passes through the dataset
batch_size = 32          # Number of samples in each minibatch
learning_rate = 0.001
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform,
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False,
                                                             extras=extras)

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = BasicCNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

#TODO: Define the loss criterion and instantiate the gradient descent optimizer
criterion = torch.nn.MultiLabelSoftMarginLoss() #TODO - loss criteria are defined in the torch.nn package

#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate) #TODO - optimizers are defined in the torch.optim package


# In[ ]:


# Track the loss across training
total_loss = []
avg_minibatch_loss = []
validation_loss = []
# Begin training procedure
for epoch in range(num_epochs):

    N = 50
    N_minibatch_loss = 0.0

    # Get the next minibatch of images, labels for training
    model.train()
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):

        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss

        #TODO: Implement cross-validation

        if minibatch_count % N == 0:

            # Print the loss averaged over the last N mini-batches
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss))

            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0

    print("Finished", epoch + 1, "epochs of training")

    #Validation
    temp_validation = 0
    model.eval()
    for minibatch_count,(images,labels) in enumerate(val_loader,0):
        images,labels = images.to(computing_device),labels.to(computing_device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        temp_validation += loss.item()

    print("Validation loss after ",epoch," epochs=",temp_validation)
    validation_loss.append(temp_validation)
    if epoch >= 1 and validation_loss[-1] > validation_loss[-2]:
        break

print("Training complete after", epoch, "epochs")

torch.save(model.state_dict(),"baseline.pt")
# In[ ]:





# In[ ]:




