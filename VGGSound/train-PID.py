import sys
import warnings
warnings.filterwarnings("ignore")
import argparse
import os, math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset.VGGSoundDataset import VGGSound
from models.basic_model import AVNet
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', default=1, type=int)

    
    parser.add_argument('--batch_size', default=128, type=int)
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
        spec = bag[0].to(device)
        image = bag[1].to(device)
        label = bag[2].to(device)

        optimizer.zero_grad()
        out_mm, out_m1, out_m2 = net(spec.unsqueeze(1).float(), image.float())

        loss1 = criterion(out_mm, label)
        loss2 = criterion(out_m1, label)
        loss3 = criterion(out_m2, label)
        loss = loss1 + loss2 + loss3
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
    n_classes = 310
    _loss = 0
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]

    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=len(test_dataloader), desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, (spec, image, label) in enumerate(test_dataloader):
        with torch.no_grad():
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out_mm, out_m1, out_m2 = net(spec.unsqueeze(1).float(), image.float())

            loss_mm = criterion(out_mm, label)
            loss_m1 = criterion(out_m1, label)
            loss_m2 = criterion(out_m2, label)
            loss = loss_mm + loss_m1 + loss_m2
            _loss += loss.item()

            out = out_mm + out_m1 + out_m2

            probs = torch.nn.functional.softmax(out, dim=1)
            preds = torch.max(probs, 1)[1]  

            correct = (preds == label).float()
            for i in range(len(label)):
                num[label[i].item()] += 1
                acc[label[i].item()] += correct[i].item()

            all_preds.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)


    pbar.close()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mAP = 0.0
    for i in range(n_classes):   
        label_binary = (all_labels == i).astype(int)
        mAP += average_precision_score(label_binary, all_preds[:, i])
    mAP /= n_classes

    accuracy = sum(acc) / sum(num)

    return _loss / epoch_step_test, accuracy, mAP

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


# Random Curriculum
def random_curriculum(dataset):

    N=len(dataset)
    idx=np.random.permutation(N)

    return (
        idx[:N//3],
        idx[N//3:2*N//3],
        idx[2*N//3:]
    )

def compute_pid_scores(train_dataset, net, device, args):

    confA_list = []
    confV_list = []
    confAV_list = []

    kl_av_list = []
    kl_va_list = []
    kl_avg_av_list = []

    net.eval()
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)

    with torch.no_grad():

        for step, bag in enumerate(dataloader):

            spec = bag[0].to(device)
            image = bag[1].to(device)
            label = bag[2].to(device)

            out_mm, out_m1, out_m2 = net(spec.unsqueeze(1).float(), image.float())

            pA = F.softmax(out_m1, dim=1)
            pV = F.softmax(out_m2, dim=1)
            pAV = F.softmax(out_mm, dim=1)

            confA = pA.gather(1, label.unsqueeze(1)).squeeze(1)
            confV = pV.gather(1, label.unsqueeze(1)).squeeze(1)
            confAV = pAV.gather(1, label.unsqueeze(1)).squeeze(1)

            confA_list.extend(confA.cpu().numpy())
            confV_list.extend(confV.cpu().numpy())
            confAV_list.extend(confAV.cpu().numpy())

            kl_av = kl_divergence(pA, pV)
            kl_va = kl_divergence(pV, pA)

            pAvg = 0.5 * (pA + pV)
            kl_avg_av = kl_divergence(pAvg, pAV)

            kl_av_list.extend(kl_av.cpu().numpy())
            kl_va_list.extend(kl_va.cpu().numpy())
            kl_avg_av_list.extend(kl_avg_av.cpu().numpy())

    confA = np.array(confA_list)
    confV = np.array(confV_list)
    confAV = np.array(confAV_list)

    kl_av = np.array(kl_av_list)
    kl_va = np.array(kl_va_list)
    kl_avg_av = np.array(kl_avg_av_list)

    # ----- PID formulas -----

    kl_sym = (kl_av + kl_va) / 2

    R = (confA * confV) * np.exp(-kl_sym)

    U = (
        confA * (1 - confV) * (1 - np.exp(-kl_av)) +
        confV * (1 - confA) * (1 - np.exp(-kl_va))
    )

    S = (confAV - np.maximum(confA, confV)) * (1 - np.exp(-kl_avg_av))

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


def build_loader(dataset, sampling_probs, batch_size):

    sampled_indices = torch.multinomial(torch.tensor(sampling_probs), num_samples=len(dataset), replacement=False)

    sampled_dataset = torch.utils.data.Subset(dataset, sampled_indices)

    current_dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

    return current_dataloader



if __name__ == '__main__':
    args = get_arguments()
    print(args)

    train_dataset = VGGSound(args, mode='train')
    test_dataset = VGGSound(args, mode='test')

    
    print('Training Data Size: ', len(train_dataset))
    print('Testing Data Size: ', len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)


    epoch_step_train = len(train_dataset) // train_dataloader.batch_size
    epoch_step_test = math.ceil(len(test_dataset) / test_dataloader.batch_size)  


    setup_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    net = AVNet(args)
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

        for epoch in range(args.epochs):
            if epoch % 5 == 0:
                        
                R, U, S = compute_pid_scores(train_dataset, net, device, args)

                pR, pU, pS = compute_pid_ratios(R, U, S)


                if epoch  < args.epochs // 3:
                    sampling_probs = pR / pR.sum()
                            

                elif epoch  < 2 * args.epochs // 3:
                    weights = 1 - pU
                    sampling_probs = weights / weights.sum()
                            

                else:
                    weights = 1 - pS
                    sampling_probs = weights / weights.sum()
                            

                current_dataloader = build_loader(
                    train_dataset,
                    sampling_probs,
                    args.batch_size
                )

            mean_loss_train = train(args, epoch, net, device, current_dataloader, optimizer, scheduler, epoch_step_train)
            total_grad_updates += len(current_dataloader)
            mean_loss_test, test_acc, test_map = test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test)


            print('********************************************************************')
            print('Epoch:' + str(epoch + 1) + '/' + str(args.epochs))
            print('Now train_loss: %.4f || Now test_loss: %.4f' % (mean_loss_train, mean_loss_test))
            print('Now test_acc: %.4f || Now test_map: %.4f' % (test_acc, test_map))


            if test_acc >= best_acc:

                best_acc = float(test_acc)
                best_acc_epoch = epoch + 1

            print('Best Test Accuracy: %.4f, Best Epoch: %d' % (best_acc, best_acc_epoch))
            print('********************************************************************')

    print("Total Gradient Updates:",total_grad_updates)
