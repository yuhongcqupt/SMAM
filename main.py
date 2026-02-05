import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3 '
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model import get_model
import evaluation
import torch
import numpy as np
from myloss import Loss
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler


def BCE_loss(target_pre, sub_target):
    return torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
                                 + (1 - sub_target).mul(torch.log(1 - target_pre + 1e-10)))))


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    # print(n)
    # print(m)
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# 计算sample 视图之间、标签之间的相似度
def view_sim(x, labels):
    n = x.size(0)
    v = x.size(1)
    if n == 1:
        return 0

    # labels = torch.matmul(label, label.T).fill_diagonal_(0)

    # 特征相似度：1）标准化 2）转换
    x = F.normalize(x, p=2, dim=-1)
    x = x.transpose(0, 1)  # [v,n,d]
    x_T = torch.transpose(x, -1, -2)  # [v,d,n]
    # feature embedding 相似度
    sim = (1 + torch.matmul(x, x_T)) / 2  # [v, n, n], 将取值范围从[-1, 1]映射到[0, 1]

    sim = sim.masked_fill(torch.eye(n,device=x.device)==1,0.)
    loss = BCE_loss(sim.view(v, -1), labels.view(1, n * n).expand(v, -1))


    return 0.5 * loss / v  # 上下三角有重复

# 计算标签的相似度Jaccard Similarity(行）
def cal_label_sim(label):
    labels = label.bool()
    intersection = (labels.unsqueeze(1) & labels.unsqueeze(0)).sum(dim=2).float()
    union = (labels.unsqueeze(1) | labels.unsqueeze(0)).sum(dim=2).float()
    jaccard_similarity = intersection / (union + 1e-9)

    return jaccard_similarity.fill_diagonal_(0)



# cosin similarity (行）
def cal_cos_sim(data):
    dim = data.shape[1]
    # 将数据转置，使得每一列变成一行
    data_T = data.T  # 维度变为 [d_model, B]

    # 计算每一行（原始数据的每一列）的 L2 范数
    norm_data_T = torch.norm(data_T, dim=1, keepdim=True)  # 维度保持为 [d_model, 1]
    # 避免分母为零，将零范数替换为一个小的正数（例如 1e-10）
    epsilon = 1e-10
    norm_data_T_safe = torch.clamp(norm_data_T, min=epsilon)

    # 计算余弦相似度矩阵
    # 使用广播机制计算每一行与每一行之间的点积，然后除以各自的范数乘积
    cosine_similarity_matrix = (data_T @ data_T.T) / (norm_data_T_safe @ norm_data_T_safe.T)
    # [-1, 1] -> [0, 1]
    cosine_similarity_matrix = (1 + cosine_similarity_matrix) / 2

    # 由于我们计算的是相似度矩阵，其对角线元素（自身与自身的相似度）为1
    # 如果需要，可以将对角线元素设为0或其他值（但通常余弦相似度中保留1是合理的）
    cosine_similarity_matrix[torch.eye(dim, dtype=torch.bool).cuda()] = 0  # 如果在GPU上运行，使用.cuda()
    # cosine_similarity_matrix[torch.eye(dim, dtype=torch.bool)] = 0  # 在CPU上运行
    return cosine_similarity_matrix


# 计算模型输出和标签（单）之间的相似度（cosine similarity )（列）
def label_sim_loss(pred, label):
    # 标签数
    l = pred.shape[1]
    pred_sim = cal_cos_sim(pred).view(1, l*l)
    # label_sim = cal_cos_sim(label).view(1, l*l)
    # 计算对应元素的差
    # difference = pred_sim - label.view(1, l*l)
    # 计算均方误差
    # mse = 0.5 * torch.mean(difference ** 2)

    loss = 0.5 * BCE_loss(pred_sim, label.view(1, l*l))  # TODO loss 可能存在nan
    return loss


def train(loader, model, loss_model, opt, sche, epoch, logger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    individual_all_z = []
    for i, (data, label) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda') for v_data in data]
        label = label.to('cuda')

        pred, x_tran, barlow_twins = model(data)
        # torch.distributed.all_reduce(x_tran)

        cls_loss2 = loss_model.BCE_loss(pred[1],label.cuda())
        # cls_loss_m2l = loss_model.BCE_loss(x_tran, label.cuda())

        # loss_ssl
        on_diag = torch.diagonal(barlow_twins).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(barlow_twins).pow_(2).sum()
        on_diag_norm = torch.sqrt(on_diag) / x_tran.shape[0]
        off_diag_norm = torch.sqrt(off_diag) / (x_tran.shape[0] * x_tran.shape[0])
        bt_loss = on_diag_norm + 0.05 * off_diag_norm

        loss = cls_loss2 + 0.1 * bt_loss


        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if isinstance(sche, CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        
        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))

    return losses, model


def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.to('cuda') for v_data in data]

        pred, _, _ = model(data)
        pred = pred[0].cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)

    evaluation_results = evaluation.do_metric(total_preds, total_labels)  # TODO 核验评估指标
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        ap=evaluation_results[0], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results


def main(args, file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir, args.name+args.dataset+'_V_'  +
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)

    # ToDo 重新编码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    for fold_idx in range(folds_num):
        fold_idx=fold_idx

        train_dataloder, train_dataset = MLdataset.getComDataloader(data_path, training_ratio=args.training_sample_ratio, val_ratio=0.15, mode='train', batch_size=args.batch_size, num_workers=4)
        val_dataloder, val_dataset = MLdataset.getComDataloader(data_path, training_ratio=args.training_sample_ratio, mode='val', batch_size=args.batch_size, num_workers=4)
        test_dataloder, test_dataset = MLdataset.getComDataloader(data_path, training_ratio=args.training_sample_ratio, mode='val', batch_size=args.batch_size, num_workers=4)

        d_list = train_dataset.d_list

        model = get_model(args, len(d_list), d_list, d_model=512, n_layers=1, heads=4, classes_num=train_dataset.classes_num, dropout=0.3, exponent=args.gamma)



        loss_model = Loss()
        # crit = nn.BCELoss()
        # optimizer_name = 'SGD'
        # lr = 1e-2
        # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr,
        #                                            weight_decay=1e-2)  # 创建优化器对象
        # scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, warmup_t=10, warmup_lr_init=5e-6)

        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

        # optimizer = Adam(model.parameters(), lr=args.lr)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.85)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
        scheduler = None

        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))

        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch = 0
        best_model_dict = {'model': model.state_dict(), 'epoch': 0}
        for epoch in range(args.epochs):
            
            train_losses, model = train(train_dataloder, model, loss_model, optimizer, scheduler, epoch, logger)
            val_results = test(val_dataloder, model, loss_model, epoch, logger)

            if val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.25>=static_res:
                static_res = val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.25
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch=epoch



            train_losses_last = train_losses
            total_losses.update(train_losses.sum)
        # 测试

        model.load_state_dict(best_model_dict['model'])
        test_results = test(test_dataloder, model, loss_model, epoch, logger)

        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'
                    .format(fold_idx, best_epoch, test_results[0], test_results[1], test_results[2], test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])

        if args.save_curve: # 未执行
            np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals,train_losses.vals))))

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP HL RL AUCme one_error coverage macAUC macro_f1 micro_f1 lr alpha beta gamma\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg,4))+'+'+str(round(res.std,4)) for res in folds_results]
    # todo 可删除
    res_list.extend([str(args.lr),str(args.alpha),str(args.beta),str(args.gamma)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()
        

def filterparam(file_path, index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH', 
                        default='/home8T/meiqiuyu/data/multi-view multi-label/')
    parser.add_argument('--dataset', type=str, default='corel5k') # mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=1, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='10_final_')


    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=150)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)


    
    args = parser.parse_args()
    
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)


    lr_list = [1e-1]  # todo 调优


    alpha_list = [1e1] # [1e0,1e1,1e2,1e-1,1e-2,1e-3] #1e2 for pascal07  1e1 for others  #
    beta_list = [1e-1]
    gamma_list = [2]

    
    for lr in lr_list:
        args.lr = lr
        if args.lr >= 0.01:
            args.momentumkl = 0.90
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma

                    for dataset in args.datasets:
                        args.dataset = dataset
                        file_path = osp.join(args.records_dir, args.name+args.dataset + '_Trai1ni1ng_i1ew_1m3v_bt1cor5k_tr_enc1_ICCV_de_view_sim_mask_11dot1test' + str(args.beta) + '_' +
                                        str(args.training_sample_ratio) + '_lr_' + str(args.lr) + '_bs128.txt')
                        args.file_path = file_path
                        existed_params = filterparam(file_path, [-3, -2, -1])
                        if [args.alpha, args.beta, args.gamma] in existed_params:
                            print('existed param! alpha:{} beta:{} gamma:{} '.format(args.alpha,args.beta,args.gamma))
                            continue
                        main(args, file_path)