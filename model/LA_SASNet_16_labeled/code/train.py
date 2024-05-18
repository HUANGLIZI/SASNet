import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from dataloaders.brats2019 import (BraTS2019, RandomCrop, ToTensor,
                                   TwoStreamBatchSampler)
#from networks.net_factory import net_factory
from networks.SASNet import SASNet
import pickle
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen
def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        #print('posmask',posmask)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)+10**(-9)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)+10**(-9))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
    return normalized_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--model_type', type=str,  default='SASNet', help='is_pretrained')
parser.add_argument('--exp', type=str,  default='SASNet', help='exp_name')
parser.add_argument('--max_iteration', type=int,  default=40000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--alpha', type=float, default=0.5, help='weight of pred')
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_labeled".format(args.dataset_name, args.exp, args.labelnum)
pretrain_path = args.root_path + "pretrain/{}_{}_{}_labeled".format(args.dataset_name, args.exp, args.labelnum)
num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = '/cpfs01/user/lizihan/lzh/diffusion/home/sdd/DTC-master/data'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path =  './data/Pancreas_CT/'
    args.max_samples = 62
elif args.dataset_name == "Brats":
    patch_size = (96, 96, 96)
    args.root_path =  './data/BraTS2019/'
    args.max_samples = 250
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    if args.model_type == 'SASNet':
        model =SASNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda() 
    elif args.model_type == 'SASNet_pretrain':
        model =SASNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda() 
        save_mode_path = os.path.join(pretrain_path, '{}_best_model.pth'.format(args.exp))
        model.load_state_dict(torch.load(save_mode_path), strict=False)
        print("init weight from {}".format(save_mode_path))
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomCrop(patch_size),
                            RandomView(args.dataset_name),
                            ToTensor(),
                            ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          RandomView(args.dataset_name),
                          ToTensor(),
                          ]))
    elif args.dataset_name == "Brats":
        db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomCrop(patch_size),
                             RandomView(args.dataset_name),
                             ToTensor(),
                         ]))
    labelnum = args.labelnum  
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    torch.autograd.set_detect_anomaly(True)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    prob_matrix={}
    updated_matrix={}
    conf_matrix={}
    model=model.to('cuda')
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, names, view, location,imgsize = sampled_batch['image'], sampled_batch['label'], sampled_batch['name'], sampled_batch['view'], sampled_batch['location'],sampled_batch['img_size']
            #location[3,4]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            model.train()
            output= model(volume_batch)
            outputs=output[0:2]
            outputs_tanhs=output[2:4]
            num_outputs = len(outputs)
            y_ori = torch.zeros((num_outputs,) + outputs[0].shape).to('cuda')
            y_wei = torch.zeros((num_outputs,) + outputs[0].shape).to('cuda')
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape).to('cuda')
            conf_matrix_patch=torch.zeros((labeled_bs,num_outputs,)+patch_size).to('cuda')
            
            for i in range(labeled_bs):
                if names[i] not in conf_matrix:
                    updated_matrix[names[i]]=torch.zeros(num_outputs, imgsize[0][i],imgsize[1][i],imgsize[2][i])
                    conf_matrix[names[i]]=0.5*torch.ones(num_outputs,imgsize[0][i],imgsize[1][i],imgsize[2][i])
        
                conf_matrix_patch[i]=conf_matrix[names[i]][:,location[0][i]:location[0][i]+patch_size[0],location[1][i]:location[1][i]+patch_size[1],location[2][i]:location[2][i]+patch_size[2]].detach().to('cuda')                         
        
            loss_seg_dice = 0 
            loss_sdf_consist=0
            loss_sdf=0
            y_prob_essemble=torch.zeros(output[0].shape).to('cuda')
            
            label_mask=torch.zeros((label_batch[:labeled_bs,...].shape[0],2,label_batch[:labeled_bs,...].shape[1],label_batch[:labeled_bs,...].shape[2],label_batch[:labeled_bs,...].shape[3])).to('cuda')
            mask_0=[label_batch[:labeled_bs,...]==0]  
            label_mask[:,0,...]=mask_0[0]
            mask_1=[label_batch[:labeled_bs,...]==1]
            label_mask[:,1,...]=mask_1[0]
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs,...]
                y_prob = F.softmax(y, dim=1)
                loss_seg_dice += dice_loss(y_prob[:,1,...], label_batch[:labeled_bs,...] == 1)
                for lb_i in range(labeled_bs):  #[2,W,H,D]
                    if args.dataset_name == "LA":
                        if view[lb_i]==0:
                            conf_matrix_patch[lb_i][idx]=conf_matrix_patch[lb_i][idx].detach().clone().permute(1,0,2)  
                    elif args.dataset_name == "Pancreas_CT" or "Brats":
                        if view[lb_i]==0:
                            conf_matrix_patch[lb_i][idx]=conf_matrix_patch[lb_i][idx].detach().clone().permute(0,2,1)
                        elif view[lb_i]==1:
                            conf_matrix_patch[lb_i][idx]=conf_matrix_patch[lb_i][idx].detach().clone().permute(2,1,0)   
                              

                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)#[B,2,W,H,D]
                y_ori[idx] = y_prob_all 
                confidences=conf_matrix_patch[:labeled_bs][idx].unsqueeze(1).repeat(1,2,1,1,1)
                y_wei[idx][:labeled_bs] =y_prob_all[:labeled_bs]*(1+args.alpha*confidences)
                conf_matrix_patch[:labeled_bs][idx]=torch.sum(y_prob_all[:labeled_bs].detach()*label_mask[:labeled_bs],dim=1)
                

                # prob_essem=torch.zeros(prob_matrix_patch[0][idx].shape).to('cuda')
                conf_essem=torch.zeros(patch_size).to('cuda')
                updated_patch=torch.zeros(patch_size).to('cuda')
                
                for lb_i in range(labeled_bs):  
                    if args.dataset_name == "LA":  
                        if view[lb_i]==0:
                            conf_matrix_patch[lb_i][idx]=conf_matrix_patch[lb_i][idx].clone().permute(1,0,2)
                    elif args.dataset_name == "Pancreas_CT" or "Brats":  
                        if view[lb_i]==0:
                            conf_matrix_patch[lb_i][idx]=conf_matrix_patch[lb_i][idx].clone().permute(0,2,1)
                            # confidence_lb[lb_i]=confidence_lb[lb_i].clone().permute(0,2,1)
                        elif view[lb_i]==1:
                            conf_matrix_patch[lb_i][idx]=conf_matrix_patch[lb_i][idx].clone().permute(2,1,0)
                            #confidence_lb[lb_i]=confidence_lb[lb_i].clone().permute(2,1,0)
                    conf_all_org=conf_matrix[names[lb_i]][idx][location[0][lb_i]:location[0][lb_i]+patch_size[0],location[1][lb_i]:location[1][lb_i]+patch_size[1],location[2][lb_i]:location[2][lb_i]+patch_size[2]].to('cuda')
                    updated_patch=updated_matrix[names[lb_i]][idx][location[0][lb_i]:location[0][lb_i]+patch_size[0],location[1][lb_i]:location[1][lb_i]+patch_size[1],location[2][lb_i]:location[2][lb_i]+patch_size[2]]
                    conf_essem[updated_patch==1]=(conf_all_org[updated_patch==1]+conf_matrix_patch[lb_i][idx][updated_patch==1])/2  
                    conf_essem[updated_patch==0]=conf_matrix_patch[lb_i][idx][updated_patch==0]

                    #update prob_matrix, confidence_matrix and updated_matrix
                    conf_matrix[names[lb_i]][idx][location[0][lb_i]:location[0][lb_i]+patch_size[0],location[1][lb_i]:location[1][lb_i]+patch_size[1],location[2][lb_i]:location[2][lb_i]+patch_size[2]]=conf_essem.detach()
                    updated_matrix[names[lb_i]][idx][location[0][lb_i]:location[0][lb_i]+patch_size[0],location[1][lb_i]:location[1][lb_i]+patch_size[1],location[2][lb_i]:location[2][lb_i]+patch_size[2]]=1           
                                      
                y_pseudo_label[idx] = sharpening(y_prob_all)
            
            y_wei_sum=torch.sum(y_wei[:,:labeled_bs],dim=0)
            y_prob_essemble[:labeled_bs]=F.softmax(y_wei_sum,dim=1)
            y_prob_essemble[labeled_bs:]=torch.mean(y_ori[:,labeled_bs:],dim = 0)
            for i in range(num_outputs):
                y_all_01=torch.where(y_prob_essemble > 0.5, torch.ones_like(y_prob_essemble), torch.zeros_like(y_prob_essemble))
                with torch.no_grad():
                    y_dis = compute_sdf(y_all_01[:,1,...].cpu(
                    ).numpy(), y_all_01[:, 0, ...].shape)
                    y_dis = torch.from_numpy(y_dis).float().cuda()         
                loss_sdf_consist += consistency_criterion(y_dis, outputs_tanhs[i][:,1,...]) 
            loss_consist = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
            
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss = args.lamda * loss_seg_dice + consistency_weight *( loss_consist+loss_sdf_consist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_d: %03f,loss_sdf_consist: %03f,loss_cosist: %03f' % (iter_num, loss, loss_seg_dice,loss_sdf_consist, loss_consist))
        
            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('Labeled_loss/loss_sdf', loss_sdf, iter_num)
            writer.add_scalar('Co_loss/loss_sdf_consist', loss_sdf_consist, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consist_weight', consistency_weight, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
            
                ins_width = 2
                B,C,H,W,D = y_ori[0].size()
                snapshot_img = torch.zeros(size = (D, 3, (num_outputs + 2) *H + (num_outputs + 2) * ins_width, W + ins_width), dtype = torch.float32)

                target =  label_batch[labeled_bs,...].permute(2,0,1)
                train_img = volume_batch[labeled_bs,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:,:, :,W:W+ins_width] = 1
                for idx in range(num_outputs+2):
                    begin_grid = idx+1
                    snapshot_img[:,:, begin_grid*H + ins_width:begin_grid*H + begin_grid*ins_width,:] = 1

                for idx in range(num_outputs):
                    begin = idx + 2
                    end = idx + 3
                    snapshot_img[:, 0, begin*H+ begin*ins_width:end*H+ begin*ins_width,:W] = y_ori[idx][labeled_bs:][0,1].permute(2,0,1)
                    snapshot_img[:, 1, begin*H+ begin*ins_width:end*H+ begin*ins_width,:W] = y_ori[idx][labeled_bs:][0,1].permute(2,0,1)
                    snapshot_img[:, 2, begin*H+ begin*ins_width:end*H+ begin*ins_width,:W] = y_ori[idx][labeled_bs:][0,1].permute(2,0,1)
                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch_num, iter_num), snapshot_img)
                
            if iter_num > 50 and iter_num % 50 == 0:
                with open(snapshot_path+'/conf_avg_iter_{}.pkl'.format(iter_num), 'wb')  as f:
                    pickle.dump(conf_matrix, f)
                model.eval()
                if args.dataset_name =="LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif args.dataset_name =="Pancreas_CT" or "Brats":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = args.dataset_name)
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.exp))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
               
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
