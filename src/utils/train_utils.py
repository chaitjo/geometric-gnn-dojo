import time
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)
        loss = F.cross_entropy(y_pred, batch.y)
        loss.backward()
        loss_all += loss.item() * batch.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def eval(model, loader, device):
    model.eval()
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            y_pred.append(model(batch).detach().cpu())
            y_true.append(batch.y.detach().cpu())
    return accuracy_score(
        torch.concat(y_true, dim=0), 
        np.argmax(torch.concat(y_pred, dim=0), axis=1)
    ) * 100  # return percentage


def _run_experiment(model, train_loader, val_loader, test_loader, n_epochs=100, verbose=True, device='cpu'):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.9, patience=25, min_lr=0.00001)
    
    if verbose:
        print(f"Running experiment for {type(model).__name__}.")
        # print("\nModel architecture:")
        # print(model)
        print(f'Total parameters: {total_param}')
        print("\nStart training:")
    
    best_val_acc = None
    perf_per_epoch = [] # Track Test/Val performace vs. epoch (for plotting)
    t = time.time()
    for epoch in range(1, n_epochs+1):
        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, optimizer, device)
        
        # Evaluate model on validation set
        val_acc = eval(model, val_loader, device)
        
        if best_val_acc is None or val_acc >= best_val_acc:
            # Evaluate model on test set if validation metric improves
            test_acc = eval(model, test_loader, device)
            best_val_acc = val_acc

        if epoch % 10 == 0 and verbose:
            print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.5f}, '
                  f'Val Acc: {val_acc:.3f}, Test Acc: {test_acc:.3f}')
        
        perf_per_epoch.append((test_acc, val_acc, epoch, type(model).__name__))
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]['lr']
    
    t = time.time() - t
    train_time = t
    if verbose:
        print(f"\nDone! Training took {train_time:.2f}s. Best validation accuracy: {best_val_acc:.3f}, corresponding test accuracy: {test_acc:.3f}.")
    
    return best_val_acc, test_acc, train_time, perf_per_epoch


def run_experiment(model, train_loader, val_loader, test_loader, n_epochs=100, n_times=100, verbose=False, device='cpu'):
    print(f"Running experiment for {type(model).__name__} ({device}).")
    
    best_val_acc_list = []
    test_acc_list = []
    train_time_list = []
    for idx in tqdm(range(n_times)):
        seed(idx) # set random seed
        best_val_acc, test_acc, train_time, _ = _run_experiment(model, train_loader, val_loader, test_loader, n_epochs, verbose, device)
        best_val_acc_list.append(best_val_acc)
        test_acc_list.append(test_acc)
        train_time_list.append(train_time)
    
    print(f'\nDone! Averaged over {n_times} runs: \n '
          f'- Training time: {np.mean(train_time_list):.2f}s ± {np.std(train_time_list):.2f}. \n '
          f'- Best validation accuracy: {np.mean(best_val_acc_list):.3f} ± {np.std(best_val_acc_list):.3f}. \n'
          f'- Test accuracy: {np.mean(test_acc_list):.1f} ± {np.std(test_acc_list):.1f}. \n')
    
    return best_val_acc_list, test_acc_list, train_time_list
