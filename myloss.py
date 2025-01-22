import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    # cosine 计算标签之间、视图之间的相似度
    def view_sim_loss2(self, x, label):

        n = x.size(0)
        v = x.size(1)
        if n == 1:
            return 0

        labels = torch.matmul(label, label.T).fill_diagonal_(0)

        # 特征相似度：1）标准化 2）转换
        x = F.normalize(x, p=2, dim=-1)
        x = x.transpose(0, 1)  # [v,n,d]
        x_T = torch.transpose(x, -1, -2)  # [v,d,n]
        # feature embedding 相似度
        sim = (1 + torch.matmul(x, x_T)) / 2  # [v, n, n], 将取值范围从[-1, 1]映射到[0, 1]

        loss = self.BCE_loss(sim.view(v, -1), labels.view(1, n * n).expand(v, -1))

        return 0.5*loss/v # 上下三角有重复


    def contrastive_loss2(self, x, inc_labels, inc_V_ind, inc_L_ind):  # x:x_transformer
        n = x.size(0)
        v = x.size(1)  # 视图数
        if n == 1:
            return 0
        valid_labels_sum = torch.matmul(inc_L_ind.float(), inc_L_ind.float().T) #[n, n] 
        # Y * G / G * G
        labels = (torch.matmul(inc_labels, inc_labels.T) / (valid_labels_sum + 1e-9)).fill_diagonal_(0)
        # labels = torch.softmax(labels.masked_fill(labels==0,-1e9),dim=-1)
        # 特征相似度：1）标准化 2）转换
        x = F.normalize(x, p=2, dim=-1)
        x = x.transpose(0, 1) #[v,n,d]
        x_T = torch.transpose(x, -1, -2)#[v,d,n]
        # feature embedding 相似度
        sim = (1+torch.matmul(x, x_T))/2 # [v, n, n]
        # 视图/模态相似度
        mask_v = (inc_V_ind.T).unsqueeze(-1).mul((inc_V_ind.T).unsqueeze(1)) #[v, n, n]
        mask_v = mask_v.masked_fill(torch.eye(n, device=x.device) == 1, 0.)
        assert torch.sum(torch.isnan(mask_v)).item() == 0
        assert torch.sum(torch.isnan(labels)).item() == 0
        assert torch.sum(torch.isnan(sim)).item() == 0
        # print('labels',torch.sum(torch.max(labels)))
        # loss = ((sim.view(v,-1)-labels.view(1,n*n))**2).mul(mask_v.view(v,-1)) # sim labels view [v, n* n]
        # 展平label计算
        loss = self.weighted_BCE_loss(sim.view(v,-1),labels.view(1,n*n).expand(v,-1),mask_v.view(v,-1),reduction='none')
        # assert torch.sum(torch.isnan(loss)).item() == 0
        
        loss =loss.sum(dim=-1)/(mask_v.view(v,-1).sum(dim=-1))
        return 0.5*loss.sum()/v   # 等式（3）


    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res
                            
    def BCE_loss(self,target_pre,sub_target):
        return torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
                                        + (1-sub_target).mul(torch.log(1 - target_pre + 1e-10)))))

