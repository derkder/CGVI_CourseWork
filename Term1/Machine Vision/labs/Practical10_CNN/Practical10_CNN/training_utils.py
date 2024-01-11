import torch
import numpy as np
torch.manual_seed(42)


def train(dataloader, model, criterion,optimizer,device='cpu'):
    '''
    This routine correspond to the steps taken for every epoch in the training schedule.
    Input: dataloader, model, criterion, optimizer and device to run model on
    Return the Average Loss and Top-1 Accuracy for the images in the given batch.
    '''
     
    model.train()
    loss_list = []
    top1_list = []

    ### TODO: Fill below the commands needed to perform the training step for one epoch, looping over the data in the dataloader. 
    ### Append losses and top-1 accuracies to the respective lists.
    ### For inspiration check out https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    for data in dataloader:
        # print(data)
        inputs = data['img'].to(device)  # Get the input images and move them to the specified device
        labels = data['target'].to(device)  # Get the labels and move them to the specified device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc = accuracy(outputs, labels, topk=(1,))[0]

        loss_list.append(loss.item())
        top1_list.append(acc.item())
        
    print('Train. Prec@1 {top1:.3f}\nTrain. Loss {loss:.3f}\n' 
          .format(top1=np.mean(top1_list),loss=np.mean(loss_list)))
    
    return np.mean(loss_list),np.mean(top1_list)


@torch.no_grad()
def validate(dataloader,model,criterion,device='cpu',split='val'):
    '''
    This routine correspond to the steps taken during the validation phase of a trained model or a model under training.
    Note that the gradients should not be updated here.
    Input: dataloader, model, criterion and device to run model on and split for reporting purposes (can be val/test)
    Return the Average Loss, Top-1 Accuracy for the images in the given batch along with the predictions. Returning predictions
    is particularly useful in the test phase when we want to dig deeper in the evaluation of the model's performance.
    '''    
    # we switch to evaluate mode during the validation phase
    model.eval()
    loss_list = []
    top1_list = []
    predictions=[]
    
    ### TODO: Fill below the commands needed to perform the validation step for one epoch (looping over the data in the dataloader) 
    ### and append losses, predictions and top-1 accuracies to the respective lists.
    ### Main difference with training step is that gradients should NOT be updated (we are not optimizing here!) and return predictions.
    for data in dataloader:
        # print(data)
        inputs = data['img'].to(device)  # Get the input images and move them to the specified device
        labels = data['target'].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Calculate accuracy and store predictions
        acc = accuracy(outputs, labels, topk=(1,))[0]
        _, preds = torch.max(outputs, 1)

        loss_list.append(loss.item())
        top1_list.append(acc.item())
        predictions.extend(preds.tolist())
        
    print('{split} prec@1 {top1:.3f}\n{split} loss {loss:.3f}\n' 
          .format(split=split.capitalize(),top1=np.mean(top1_list),loss=np.mean(loss_list)))
    
    return np.mean(loss_list),np.mean(top1_list),predictions
   
def accuracy(output, targets, topk=(1,)):
    '''Computes the precision@k for the specified values of k. 
       Helper util to be used during training/validation.
       Input is model outputs and targets in tensor format'''
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res      
