import numpy as np
import torch
import math


def adjust_learning_rate(optimizer, epoch,args):
    if args.lradj == 'binary':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch) // 1))}
    elif args.lradj == 'type0':
        lr_adjust = {epoch: args.learning_rate if epoch<1 else args.learning_rate * (0.5 ** (((epoch-1)) // 1))}
    elif args.lradj == 'type05':
        lr_adjust = {epoch: args.learning_rate if epoch<5 else args.learning_rate * (0.5 ** (((epoch-4)) // 1))}
    elif args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate if epoch<10 else args.learning_rate * (0.5 ** (((epoch-9)) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate if epoch<20 else args.learning_rate * (0.5 ** (((epoch-19)) // 1))}
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch<30 else args.learning_rate * (0.5 ** (((epoch-29)) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch<40 else args.learning_rate * (0.5 ** (((epoch-39)) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Metric score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n"
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


