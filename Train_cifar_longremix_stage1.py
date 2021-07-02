from __future__ import print_function
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
# from torch.utils.tensorboard import SummaryWriter
import pdb
import io
import PIL
from torchvision import transforms
import seaborn as sns
import sklearn.metrics as metrics
import pickle
import json
import pandas as pd
import time
from pathlib import Path
from utils_plot import plot_guess_view, plot_histogram_loss_pred, plot_model_view_histogram_loss, plot_model_view_histogram_pred, plot_tpr_fpr
sns.set()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--num_clean', default=5, type=int)
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--run', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)



# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, savelog=False):
    net.train()
    net2.eval() #fix one network and train the other

    train_loss = train_loss_lx = train_loss_u = train_loss_penalty = 0
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty

        train_loss += loss
        train_loss_lx += Lx
        train_loss_u += Lu
        train_loss_penalty += penalty
         

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

    if savelog:
        train_loss /= len(labeled_trainloader.dataset)
        train_loss_lx /= len(labeled_trainloader.dataset)
        train_loss_u /= len(labeled_trainloader.dataset)
        train_loss_penalty /= len(labeled_trainloader.dataset)
        # Record training loss from each epoch into the writer
        # writer_tensorboard.add_scalar('Train/Loss', train_loss.item(), epoch)
        # writer_tensorboard.add_scalar('Train/Lx', train_loss_lx.item(), epoch)
        # writer_tensorboard.add_scalar('Train/Lu', train_loss_u.item(), epoch)
        # writer_tensorboard.add_scalar('Train/penalty', train_loss_penalty.item(), epoch)
        # writer_tensorboard.close()

def warmup(epoch,net,optimizer,dataloader,savelog=False):
    net.train()
    wm_loss = 0
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss

        wm_loss += L
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

    if savelog:
        wm_loss /= len(dataloader.dataset)
        # Record training loss from each epoch into the writer
        # writer_tensorboard.add_scalar('Warmup/Loss', wm_loss.item(), epoch)
        # writer_tensorboard.close()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)  

            test_loss += CEloss(outputs1, targets)
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                                
    acc = 100.*correct/total
    acc_hist.append(acc)
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

    # Record loss and accuracy from the test run into the writer
    test_loss /= len(test_loader.dataset)
    # writer_tensorboard.add_scalar('Test/Loss', test_loss, epoch)
    # writer_tensorboard.add_scalar('Test/Accuracy', acc, epoch)
    # writer_tensorboard.close()

def eval_train(model,all_loss, all_preds, all_hist, savelog=False):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))    
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), args.num_class)
    eval_loss = train_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            eval_loss += CEloss(outputs, targets)

            _, pred = torch.max(outputs.data, -1)
            acc = float((pred==targets.data).sum()) 
            train_acc += acc
            eval_preds = F.softmax(outputs, -1).cpu().data

            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b] 
                preds[index[b]] = eval_preds[b][targets[b]]   
                preds_classes[index[b]] =  eval_preds[b]

    if savelog:
        eval_loss /= len(eval_loader.dataset)
        train_acc /= len(eval_loader.dataset)
        # writer_tensorboard.add_scalar('eval/Loss', eval_loss.item(), epoch)
        # writer_tensorboard.add_scalar('eval/acc', train_acc, epoch)
        # writer_tensorboard.close()

    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    all_preds.append(preds)
    all_hist.append(preds_classes)


    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         

    return prob,all_loss, all_preds, all_hist

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

def guess_unlabeled(net1, net2, unlabeled_trainloader):
    net1.eval()
    net2.eval()

    guessedPred_unlabeled  = []
    for batch_idx, (inputs_u, inputs_u2) in enumerate(unlabeled_trainloader): 

        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net1(inputs_u)
            outputs_u12 = net1(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()  

            _, guessed_u = torch.max(targets_u, dim=-1)
            guessedPred_unlabeled.append(guessed_u) 

    return torch.cat(guessedPred_unlabeled)

def save_models(epoch, net1, optimizer1, net2, optimizer2, save_path):
    state = ({
                    'epoch'     : epoch,
                    'state_dict1'     : net1.state_dict(),
                    'optimizer1'      : optimizer1.state_dict(),
                    'state_dict2'     : net2.state_dict(),
                    'optimizer2'      : optimizer2.state_dict()
                    
                })
    state2 = ({'all_loss': all_loss,
                    'all_preds': all_preds,
                    'hist_preds': hist_preds,
                    'inds_clean': inds_clean,
                    'inds_noisy': inds_noisy,
                    'clean_labels': clean_labels,
                    'noisy_labels': noisy_labels,
                    'all_idx_view_labeled': all_idx_view_labeled,
                    'all_idx_view_unlabeled': all_idx_view_unlabeled,
                    'all_superclean': all_superclean
                    })
    state3 = ({
                'all_superclean': all_superclean
                })


    if epoch%1==0:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        # fn2_log = os.path.join(save_path, 'model_ckpt_hist.pth.tar')
        # torch.save(state2, fn2_log)
        #fn3 = os.path.join(save_path, 'superclean.pth.tar')
        fn3 = os.path.join('hcs/', 'hcs_%s_%.2f_%s_cn%d_run%d.pth.tar'%(args.dataset, args.r, args.noise_mode,args.num_clean, args.run))
        torch.save(state3, fn3)

        # fn4 = os.path.join(save_path, 'superclean_%s_%.2f_%s_cn%d.json'%(args.dataset, args.r, args.noise_mode,args.num_clean))
        # json.dump(all_superclean,open(fn4,"w"))


# def plot_graphs(epoch):
#     num_inds_clean = len(inds_clean)
#     num_inds_noisy = len(inds_noisy)
#     perc_clean = 100*num_inds_clean/float(num_inds_clean+num_inds_noisy)

#     plt.hist(all_loss[0][-1].numpy(), bins=20, range=(0., 1.), edgecolor='black', color='g')
#     plt.xlabel('loss');
#     plt.ylabel('number of data')
#     plt.savefig('%s/histogram_epoch%03d.png' % (path_exp,epoch))
#     # buf = io.BytesIO()
#     # plt.savefig(buf, format='png')
#     # buf.seek(0)
#     # image = PIL.Image.open(buf)
#     # image = transforms.ToTensor()(image)
#     # writer_tensorboard.add_image('Histogram/loss_all', image, epoch)
#     plt.clf()

#     plt.hist(all_loss[0][-1].numpy()[inds_clean],bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
#     if len(inds_noisy) >0:
#         plt.hist(all_loss[0][-1].numpy()[inds_noisy], bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
#     plt.xlabel('loss');
#     plt.ylabel('number of data')
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#        ncol=2, mode="expand", borderaxespad=0.)
#     plt.savefig('%s/sep_loss_epoch%03d.png' % (path_exp,epoch))
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image = PIL.Image.open(buf)
#     image = transforms.ToTensor()(image)
#     # writer_tensorboard.add_image('Histogram/loss_sep', image, epoch)
#     plt.clf()      

#     plt.hist(all_preds[0][-1].numpy()[inds_clean],bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
#     if len(inds_noisy) >0:
#         plt.hist(all_preds[0][-1].numpy()[inds_noisy], bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
#     plt.xlabel('prob');
#     plt.ylabel('number of data')
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#        ncol=2, mode="expand", borderaxespad=0.)
#     plt.savefig('%s/preds_sep_epoch%03d.jpg' % (path_exp,epoch))
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image = PIL.Image.open(buf)
#     image = transforms.ToTensor()(image)
#     # writer_tensorboard.add_image('Histogram/prob_sep', image, epoch)
#     plt.clf() 


name_exp = 'longremix_stage1_cn%d'%args.num_clean

exp_str = '%s_%.2f_%s_%s_lu_%d'%(args.dataset, args.r, args.noise_mode, name_exp, int(args.lambda_u))
if args.run >0:
    exp_str = exp_str + '_run%d'%args.run
path_exp='./checkpoint/' + exp_str
# try:
#     os.stat(path_exp)
# except:
#     os.mkdir(path_exp)

path_plot = os.path.join(path_exp, 'plots')

Path(path_exp).mkdir(parents=True, exist_ok=True)
Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
Path(path_plot).mkdir(parents=True, exist_ok=True)


incomplete = os.path.exists("./checkpoint/%s/model_ckpt.pth.tar"%(exp_str))
print('Incomplete...', incomplete)

if incomplete == False:
    stats_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
    test_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt','w') 
    time_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt','w') 
else:    
    stats_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_stats.txt','a') 
    test_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str,args.dataset,args.r,args.noise_mode)+'_acc.txt','a') 
    time_log=open('./checkpoint/%s/%s_%.2f_%s'%(exp_str, args.dataset,args.r,args.noise_mode)+'_time.txt','a') 

# writer_tensorboard = SummaryWriter('tensor_runs/'+exp_str) 
  

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='noise/%s/%.2f_%s.json'%(args.dataset,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks
all_preds = [[], []] # save the history of preds for two networks 
hist_preds = [[],[]]
all_idx_view_labeled = [[],[]]
all_idx_view_unlabeled = [[], []]

resume_epoch = 0

if incomplete == True:
    print('loading Model...\n')
    load_path = 'checkpoint/%s/model_ckpt.pth.tar'%(exp_str)
    ckpt = torch.load(load_path)
    resume_epoch = ckpt['epoch']
    print('resume_epoch....', resume_epoch)
    net1.load_state_dict(ckpt['state_dict1'])
    net2.load_state_dict(ckpt['state_dict2'])
    optimizer1.load_state_dict(ckpt['optimizer1'])
    optimizer2.load_state_dict(ckpt['optimizer2'])

test_loader = loader.run('test')
eval_loader = loader.run('eval_train') 
noisy_labels = eval_loader.dataset.noise_label
clean_labels = eval_loader.dataset.train_label 
inds_noisy = np.asarray([ind for ind in range(len(noisy_labels)) if noisy_labels[ind] != clean_labels[ind]])
inds_clean = np.delete(np.arange(len(noisy_labels)), inds_noisy)
all_superclean = [[],[]]

total_time =  0
warmup_time = 0
acc_hist = []

for epoch in range(resume_epoch, args.num_epochs+1):   
    lr=args.lr
    # if epoch >= 150:
    #     lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
         
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')

        start_time = time.time()
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader, savelog=True)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader, savelog=False) 
        end_time = round(time.time() - start_time)
        total_time+= end_time
        warmup_time+= end_time

        #save histogram

        prob1, all_loss[0], all_preds[0], hist_preds[0] = eval_train(net1, all_loss[0], all_preds[0], hist_preds[0])   
        prob2, all_loss[1], all_preds[1], hist_preds[1] = eval_train(net1, all_loss[1], all_preds[1], hist_preds[1])   

        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)
        idx_view_labeled = (pred1).nonzero()[0]
        idx_view_unlabeled = (1-pred1).nonzero()[0]
        all_idx_view_labeled[0].append(idx_view_labeled)
        all_idx_view_labeled[1].append((pred2).nonzero()[0])
        all_idx_view_unlabeled[0].append(idx_view_unlabeled)
        all_idx_view_unlabeled[1].append((1-pred2).nonzero()[0])

        if epoch==(warm_up-1):
            time_log.write('Warmup: %f \n'%(warmup_time))
            time_log.flush()  


        if epoch % 5==0:
            # plot_graphs(epoch)
            plot_histogram_loss_pred(data=all_loss[0][-1].numpy(), inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )


        
   
    else:         
        start_time = time.time()
        prob1,all_loss[0], all_preds[0], hist_preds[0] =eval_train(net1,all_loss[0], all_preds[0], hist_preds[0], savelog=True)   
        prob2,all_loss[1], all_preds[1], hist_preds[1] =eval_train(net2,all_loss[1], all_preds[1], hist_preds[1],savelog=False) 

        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)

        idx_view_labeled = (pred1).nonzero()[0]
        idx_view_unlabeled = (1-pred1).nonzero()[0]
        all_idx_view_labeled[0].append(idx_view_labeled)
        all_idx_view_labeled[1].append((pred2).nonzero()[0])
        all_idx_view_unlabeled[0].append(idx_view_unlabeled)
        all_idx_view_unlabeled[1].append((1-pred2).nonzero()[0])

         #check hist of predclean
        superclean = []
        nclean = args.num_clean
        #for ii in range(50000):
        for ii in range(len(eval_loader.dataset)):
            clean_lastn = True
            for h_ep in all_idx_view_labeled[0][-nclean:]:   #check last nclean epochs
                if ii not in h_ep:
                    clean_lastn = False
                    break
            if clean_lastn:
                superclean.append(ii)
        print('\nsuperclean: %d'%len(superclean))
        all_superclean[0].append(superclean)
        pred1 = np.array([True if p in superclean else False for p in range(len(pred1))])

         #check hist of predclean
        superclean = []
        nclean = args.num_clean
        #for ii in range(50000):
        for ii in range(len(eval_loader.dataset)):
        
            clean_lastn = True
            for h_ep in all_idx_view_labeled[1][-nclean:]:   #check last nclean epochs
                if ii not in h_ep:
                    clean_lastn = False
                    break
            if clean_lastn:
                superclean.append(ii)
        all_superclean[1].append(superclean)
        pred2 = np.array([True if p in superclean else False for p in range(len(pred2))])

        end_time = round(time.time() - start_time)
        total_time+= end_time

        if epoch%10==0:
            # plot_graphs(epoch)
            plot_histogram_loss_pred(data=all_loss[0][-1].numpy(), inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )

            idx_view_labeled = (pred1).nonzero()[0]
            idx_view_unlabeled = (1-pred1).nonzero()[0]

            

            plot_model_view_histogram_loss(data=all_loss[0][-1].numpy(), idx_view_labeled=idx_view_labeled,
             idx_view_unlabeled=idx_view_unlabeled, inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )
            
            plot_model_view_histogram_pred(data=all_preds[0][-1].numpy(), idx_view_labeled=idx_view_labeled,
             idx_view_unlabeled=idx_view_unlabeled, inds_clean=inds_clean, inds_noisy=inds_noisy, path=path_plot, epoch=epoch )

            # if len(inds_noisy) >0:
            #     plot_tpr_fpr(noisy_labels=noisy_labels, clean_labels=clean_labels, prob=prob1)

                

        start_time = time.time()
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader, _ = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,savelog=True) # train net1  
        
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader, u_map_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,savelog=False) # train net2         
        end_time = round(time.time() - start_time)
        total_time+= end_time

        if epoch%10==0:
            guessed =guess_unlabeled(net1, net2, u_map_trainloader)
            idx_unlabeled = (1-pred1).nonzero()[0] 
            inds_guess_wrong = np.asarray([idx_unlabeled[ind] for ind in range(len(idx_unlabeled)) if clean_labels[idx_unlabeled[ind]] != guessed[ind]])
            inds_guess_correct = np.asarray([idx_unlabeled[ind] for ind in range(len(idx_unlabeled)) if clean_labels[idx_unlabeled[ind]] == guessed[ind]])

            plot_guess_view(data=all_loss[0][-1].numpy(), inds_guess_correct=inds_guess_correct, inds_guess_wrong=inds_guess_wrong, path=path_plot, epoch=epoch)


    save_models(epoch, net1, optimizer1, net2, optimizer2, path_exp)

    test(epoch,net1,net2)

test_log.write('\nBest:%.2f  avgLast10: %.2f\n'%(max(acc_hist),sum(acc_hist[-10:])/10.0))
test_log.close() 

time_log.write('SSL Time: %f \n'%(total_time-warmup_time))
time_log.write('Total Time: %f \n'%(total_time))
time_log.close()

