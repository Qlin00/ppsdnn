import os
import gnn_dataloader
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from model import GNN

device = torch.device("cuda:6" if torch.cuda.is_available() else 'cpu')
target_device = "nvidia_2080ti"
print("Processing Training Set")
train_set = gnn_dataloader.GNNDataset(train=True, device=target_device)
print("Processing Testing Set")
test_set = gnn_dataloader.GNNDataset(train=False, device=target_device)
print("successful")


train_loader = gnn_dataloader.GNNDataloader(train_set,batchsize=1,shuffle=True)
test_loader = gnn_dataloader.GNNDataloader(test_set,batchsize=1,shuffle=False)
print('Train Dataset Size: ', len(train_set))
print('Test Dataset Size: ', len(test_set))
print('Attribute tensor shape:',next(train_loader)[1].ndata['h'].size(1))
ATTR_COUNT = next(train_loader)[1].ndata['h'].size(1)

MIN_MAPE = 10000
MIN_RMSE = 10000
MAX_acc = 0



def MAPE(y_true,y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true)) * 100

def test_epoch(epoch,model):
    global MIN_MAPE
    global MIN_RMSE
    global MAX_acc
    real = []
    pred = []
    test_length = len(test_set)
    test_acc_ten = 0

    for batched_l, batched_g, batched_m in test_loader:
        batched_l = batched_l.to(device).float()
        batched_g = batched_g.to(device)
        batched_f = batched_g.ndata['h'].float()
        batched_m = batched_m.to(device).float()
        logits = model(batched_g, batched_f, batched_m)
        for i in range(len(batched_l)):
            pred_latency = logits[i].item()
            prec_latency = batched_l[i].item()
            real.append(prec_latency)
            pred.append(pred_latency)
            if (pred_latency >= 0.9 * prec_latency) and (pred_latency <= 1.1 * prec_latency):
                test_acc_ten += 1
    

    Real = np.array(real)
    Pred = np.array(pred)


    rmse = np.sqrt(mean_squared_error(Pred,Real))
    mape = MAPE(Real,Pred)
    if mape < MIN_MAPE:    
        MIN_MAPE = mape
        MIN_RMSE = rmse
        MAX_acc = test_acc_ten / test_length * 100
        torch.save(model.state_dict(),"gnnModel.pt")
        np.savetxt('pred.txt',np.array(pred))
        np.savetxt('real.txt',np.array(real))

    #print("Test accuracy within 10%: ", test_acc_ten / test_length * 100," %.")
    #print("MAPE: ",mape," %.")
    #print("RMSE: ",rmse)
    
    print("[Epoch ", epoch,"]: ","Testing accuracy within 10%: ", test_acc_ten / test_length * 100," %."," MAPE: ",mape," %."," RMSE: ",rmse)

from argparse import ArgumentParser
def main():
    # parser = ArgumentParser()
    # parser.add_argument('--model_name', default='VGG')
    # parser.add_argument('--learning_rate', default=4e-4)
    # parser.add_argument('--batch_size', default=8, type=int)
    # parser.add_argument('--output', default='output.csv')

    # args = parser.parse_args()
 

    from torch.optim.lr_scheduler import CosineAnnealingLR

    if torch.cuda.is_available():
        print("Using CUDA")

    model = GNN(ATTR_COUNT, 3,400,0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=4e-4)

    EPOCHS=200
    loss_func = nn.MSELoss() 

    real = []
    pred = []


    lr_scheduler = CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_sum = 0
    for epoch in range(EPOCHS):
        train_length = len(train_set)
        train_acc_ten = 0
        loss_sum = 0

        for batched_l, batched_g, batched_m in train_loader:
            opt.zero_grad()
            batched_l = batched_l.to(device).float()
            batched_g = batched_g.to(device)
            batched_f = batched_g.ndata['h'].float()
            batched_m = batched_m.to(device).float()
            logits = model(batched_g, batched_f, batched_m)
            for i in range(len(batched_l)):
                pred_latency = logits[i].item()
                prec_latency = batched_l[i].item()
                if (pred_latency >= 0.9 * prec_latency) and (pred_latency <= 1.1 * prec_latency):
                    train_acc_ten += 1

            batched_l = torch.reshape(batched_l,(-1,1))
            loss = loss_func(logits, batched_l)
            loss_sum += loss
            loss.backward()
            opt.step()
        lr_scheduler.step()
        print("[Epoch ", epoch,"]: ","Training accuracy within 10%: ", train_acc_ten / train_length * 100," %.")
        test_epoch(epoch,model)

    print("[Training done] " ,"Testing accuracy within 10%: ", MAX_acc," %."," MAPE: ",MIN_MAPE," %."," RMSE: ",MIN_RMSE)


if __name__ == '__main__':
    main()

