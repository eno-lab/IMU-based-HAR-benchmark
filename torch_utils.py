import torch 
import torch.utils.data as Data
import numpy as np

def calc_loss_acc_output(model, loss_function, x_data, y_data, time_callback=None):
    """
       calculate loss, acc, output 
    """
    loss_sum_data = torch.tensor(0)
    output = []
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = x_data.shape[0],
                                  shuffle = False)

    if time_callback is not None:
        time_callback.on_test_begin()
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            if time_callback is not None:
                time_callback.on_test_batch_begin()
            output_bc = model(x)
            if time_callback is not None:
                time_callback.on_test_batch_end()
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
                
            loss_bc = loss_function(output_bc, y)
            loss_sum_data = loss_sum_data + loss_bc

            output.extend(output_bc.cpu().data.numpy())
    
    loss = loss_sum_data.data.item()/y_data.shape[0]
    output = np.array(output).argmax(axis=1)
    acc = np.sum(output == y_data.argmax(1))/y_data.shape[0]
    #acc_train = true_sum_data.data.item()/_y_test.shape[0]
    
    return loss, acc, output 

