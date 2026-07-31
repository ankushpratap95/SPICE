import sys
import warnings
warnings.filterwarnings("ignore")
import argparse
import os, math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset.nvGesture import NvGestureDataset
from models.RODModel import TriNet
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, accuracy_score

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='./data/nvGesture/nvGesture_v1/', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--warm_up', default=30, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--random_seed', default=1751780633, type=int)
    return parser.parse_args()

def train(args, epoch, net, device, train_dataloader, optimizer, scheduler, epoch_step_train):

    criterion = nn.CrossEntropyLoss()
    _total_loss = 0

    epochs = args.warm_up if scheduler is None else args.epochs

    print('Start Training')
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.train()

    for step, bag in enumerate(train_dataloader):
        rgb = bag[0].float().to(device)
        of = bag[1].to(device)
        depth = bag[2].to(device)
        label = bag[3].to(device)

        optimizer.zero_grad()
        out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

        loss1 = criterion(out_mm, label)
        loss2 = criterion(out_m1, label)
        loss3 = criterion(out_m2, label)
        loss4 = criterion(out_m3, label)
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()

        _total_loss += loss.item()

        pbar.set_postfix(**{'train_loss': _total_loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
        pbar.update(1)
    pbar.close()
    if scheduler:
        scheduler.step() 

    return _total_loss / len(train_dataloader)

def test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    _loss = 0

    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=len(test_dataloader), desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, bag in enumerate(test_dataloader):
        with torch.no_grad():
            rgb = bag[0].float().to(device)
            of = bag[1].to(device)
            depth = bag[2].to(device)
            label = bag[3].to(device)

            out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

            loss1 = criterion(out_mm, label)
            loss2 = criterion(out_m1, label)
            loss3 = criterion(out_m2, label)
            loss4 = criterion(out_m3, label)
            loss = loss1 + loss2 + loss3 + loss4
            _loss += loss.item()

            out = out_m1 + out_m2 + out_m3 + out_mm
            preds = torch.max(out, dim=1)[1]  
            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')

    return _loss / epoch_step_test, accuracy, f1_macro


def kl_divergence(p, q, eps=1e-8):
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=1)


def normalize_min_max(x):
    x = np.array(x)
    denom = np.max(x) - np.min(x)
    if denom == 0:
        return np.zeros_like(x)
    return (x - np.min(x)) / denom

def compute_pid_scores(train_dataset, net, device, args):

    conf1_list = []
    conf2_list = []
    conf3_list = []
    confMM_list = []

    kl_r1_list = []
    kl_r2_list= []
    kl_r3_list = []
    kl_syn_list = []
    
    net.eval()
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)

    with torch.no_grad():

        for step, bag in enumerate(dataloader):

            rgb = bag[0].float().to(device)
            of = bag[1].to(device)
            depth = bag[2].to(device)
            label = bag[3].to(device)

            out_m1, out_m2, out_m3, out_mm = net(rgb, of, depth)

            # Softmax probabilities
            p1 = F.softmax(out_m1, dim=1)
            p2 = F.softmax(out_m2, dim=1)
            p3 = F.softmax(out_m3, dim=1)
            pMM = F.softmax(out_mm, dim=1)

            # True-class confidences
            conf1 = p1.gather(1, label.unsqueeze(1)).squeeze(1)
            conf2 = p2.gather(1, label.unsqueeze(1)).squeeze(1)
            conf3 = p3.gather(1, label.unsqueeze(1)).squeeze(1)
            confMM = pMM.gather(1, label.unsqueeze(1)).squeeze(1)

            conf1_list.extend(conf1.cpu().numpy())
            conf2_list.extend(conf2.cpu().numpy())
            conf3_list.extend(conf3.cpu().numpy())
            confMM_list.extend(confMM.cpu().numpy())

            # other-modality distributions
            p_rest_1 = (p2 + p3) / 2
            p_rest_2 = (p1 + p3) / 2
            p_rest_3 = (p1 + p2) / 2

            # divergence from rest
            kl_r1 = kl_divergence(p1, p_rest_1)
            kl_r2 = kl_divergence(p2, p_rest_2)
            kl_r3 = kl_divergence(p3, p_rest_3)

            kl_r1_list.extend(kl_r1.cpu().numpy())
            kl_r2_list.extend(kl_r2.cpu().numpy())
            kl_r3_list.extend(kl_r3.cpu().numpy())


            p_uni_cons = (p1 + p2 + p3) / 3
            kl_syn = kl_divergence(pMM, p_uni_cons)
            kl_syn_list.extend(kl_syn.cpu().numpy())

    conf1 = np.array(conf1_list)
    conf2 = np.array(conf2_list)
    conf3 = np.array(conf3_list)
    confMM = np.array(confMM_list)

    kl_r1 = np.array(kl_r1_list)
    kl_r2 = np.array(kl_r2_list)
    kl_r3 = np.array(kl_r3_list)
    kl_syn = np.array(kl_syn_list)

    # ----- PID formulas -----

    kl_r = (kl_r1 + kl_r2 + kl_r3) / 3

    R = (conf1 * conf2 * conf3) * np.exp(-kl_r)

    U1 = conf1 * (1 - conf2) * (1 - conf3) * (1 - np.exp(-kl_r1))
    U2 = conf2 * (1 - conf1) * (1 - conf3) * (1 - np.exp(-kl_r2)) 
    U3 = conf3 * (1 - conf1) * (1 - conf2) * (1 - np.exp(-kl_r3))

    U = U1 + U2 + U3

    best_uni = np.maximum.reduce([conf1, conf2, conf3])
    S = (confMM - best_uni) * (1 - np.exp(-kl_syn))
    # S = np.maximum(confMM - best_uni, 0) * (1 - np.exp(-kl_syn))

    R = normalize_min_max(R)
    U = normalize_min_max(U)
    S = normalize_min_max(S)

    return R, U, S

def compute_pid_ratios(R, U, S):

    total = R + U + S + 1e-8

    pR = R / total
    pU = U / total
    pS = S / total

    return pR, pU, pS

def classify_pid_samples(pR, pU, pS):

    pid_scores = np.stack([pR, pU, pS], axis=1)

    pid_class = np.argmax(pid_scores, axis=1)

    r_idx = np.where(pid_class == 0)[0]
    u_idx = np.where(pid_class == 1)[0]
    s_idx = np.where(pid_class == 2)[0]

    return r_idx, u_idx, s_idx

def sort_pid_groups(r_idx, u_idx, s_idx, pR, pU, pS):

    # R: strongest redundancy first
    r_sorted = r_idx[np.argsort(-pR[r_idx])]



    # U: easier uniqueness first
    u_sorted = u_idx[np.argsort(pU[u_idx])]


    # S: easier synergy first
    s_sorted = s_idx[np.argsort(pS[s_idx])]
 

    curriculum_idx = np.concatenate([r_sorted, u_sorted, s_sorted])


    return r_sorted, u_sorted, s_sorted


def get_stage_indices(epoch, T, r_sorted, u_sorted, s_sorted):

    if epoch < T // 3:

        active_idx = r_sorted

    elif epoch < 2 * T // 3:

        active_idx = np.concatenate([r_sorted, u_sorted])

    else:

        active_idx = np.concatenate([r_sorted, u_sorted, s_sorted])

    return active_idx


def build_loader(dataset, indices, batch_size):

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32, 
        pin_memory=True, 
        drop_last=True
    )

    return loader



if __name__ == '__main__':
    args = get_arguments()
    print(args)

    train_dataset = NvGestureDataset(args, mode='train')
    test_dataset = NvGestureDataset(args, mode='test')

    
    print('Training Data Size: ', len(train_dataset))
    print('Testing Data Size: ', len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)


    epoch_step_train = len(train_dataset) // train_dataloader.batch_size
    epoch_step_test = math.ceil(len(test_dataset) / test_dataloader.batch_size)  

    setup_seed(args.random_seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    net = TriNet(args)
    net.apply(weight_init)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.epochs,  # Number of iterations for cosine cycle
    #     eta_min=0.1 * args.learning_rate  # Minimum learning rate
    # )

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    
    # Warmup Epochs
    for epoch in range(args.warm_up):
                mean_loss_train = train(args, epoch, net, device, train_dataloader, optimizer, scheduler=None, epoch_step_train=epoch_step_train)
                print('********************************************************************')
                print('Warm Up Epoch:' + str(epoch + 1) + '/' + str(args.warm_up))
                print('Now train_loss: %.4f' % (mean_loss_train))
                print('********************************************************************')

    if True:
        best_acc = 0.0
        best_acc_epoch = 0
        total_grad_updates = 0
        current_dataloader = train_dataloader
        sample_count_history = {
            "epoch": [],
            "R": [],
            "U": [],
            "S": [],
            "total": []
            }
        
        test_accuracies = []
        gradient_updates = []

        for epoch in range(args.epochs):
            if epoch % 5 == 0:
                        
                R, U, S = compute_pid_scores(train_dataset, net, device, args)

                pR, pU, pS = compute_pid_ratios(R, U, S)

                r_idx, u_idx, s_idx = classify_pid_samples(pR, pU, pS)

                r_count = len(r_idx)
                u_count = len(u_idx)
                s_count = len(s_idx)

                sample_count_history["epoch"].append(epoch)
                sample_count_history["R"].append(r_count)
                sample_count_history["U"].append(u_count)
                sample_count_history["S"].append(s_count)
                sample_count_history["total"].append(r_count + u_count + s_count)

                r_sorted, u_sorted, s_sorted = sort_pid_groups(
                    r_idx, u_idx, s_idx, pR, pU, pS
                    )

                active_idx = get_stage_indices(
                    epoch,
                    args.epochs,
                    r_sorted,
                    u_sorted,
                    s_sorted
                    )

                current_dataloader = build_loader(
                    train_dataset,
                    active_idx,
                    args.batch_size
                    )

            mean_loss_train = train(args, epoch, net, device, current_dataloader, optimizer, scheduler, epoch_step_train)
            total_grad_updates += len(current_dataloader)
            mean_loss_test, test_acc, test_f1 = test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test)
            test_accuracies.append(test_acc)
            gradient_updates.append(total_grad_updates)

            print('********************************************************************')
            print('Epoch:' + str(epoch + 1) + '/' + str(args.epochs))
            print('Now train_loss: %.4f || Now test_loss: %.4f' % (mean_loss_train, mean_loss_test))
            print('Now test_acc: %.4f || Now test_f1: %.4f' % (test_acc, test_f1))


            if test_acc >= best_acc:
                best_acc = float(test_acc)
                best_acc_epoch = epoch + 1

            print('Best Test Accuracy: %.4f, Best Epoch: %d' % (best_acc, best_acc_epoch))
            print('********************************************************************')

    print("Total Gradient Updates:",total_grad_updates)
    print('Sample Count History:', sample_count_history)
    print('Test Accuracies:', test_accuracies)
    print('Gradient Updates:', gradient_updates)
