import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch.nn as nn
from skimage.morphology import skeletonize
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch
import numpy as np
import torch.nn.functional as F
class gwDistance(nn.Module):
    def __init__(self,eps=0.1, thresh=0.1, max_iter=100,reduction='none'):
        super(gwDistance,self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.thresh = thresh
        self.mask_matrix = None

    def forward(self,pred,target):
        """
        :param pred: torch.tensor, with size (batch,channel,x,y,z) or (b,n,2）
        :param target: same with 1.
        1.Build Graph Where Pred>0.5 TODO:Save mask graph in Hard Drive
        2.Calculate Cost Matrix basing on Euclidean distance
        3.Calculate Mask Matrix Basing on k-nn strategy
        4.Calculate Optimal Plan using Sinkhorn Iteration Algorithm
        :return:
        """
        assert pred.size() == target.size()
        batch = pred.size(0)
        gwd=0
        for i in range (batch):# batch_size
            # channel=1
            pred[i,0] = self.extract_centerline_2d(pred[i,0])
            target[i,0] = self.extract_centerline_2d(target[i,0], is_gt=True)

            source_graph = self.build_graph_from_torch(pred[i,0]).cuda()
            mask_graph = self.build_graph_from_torch(target[i,0], is_gt=True).cuda()

            cost_matrix = self.get_cost_matrix(source_graph, mask_graph).cuda()  # （source_graph_node_num,mask_graph_node_num)
            mask_matrix = self.get_mask_matrix(cost_matrix).to_sparse().cuda() # 二值的（source_graph_node_num,mask_graph_node_num)

            mu = self.marginal_prob_uniform(pred[i,0])
            nu = self.marginal_prob_uniform(target[i,0])
            distance = self.get_dist(mu, nu, cost_matrix, mask_matrix)
            # print(distance)
            gwd = gwd + distance

        gwd = gwd / batch
        return gwd

    def get_dist(self,mu,nu,C,A,mask=None):
        """
        :param mu: 预测图的分布
        :param nu: GT的分布
        :param C: cost 矩阵
        :param A: mask矩阵（k近邻，GTOT里的邻接矩阵）
        :param mask: （未知）
        :return:
        """
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        if A is not None:
            if A.type().startswith('torch.cuda.sparse'):
                self.sparse = True
                C = A.to_dense() * C
            else:
                self.sparse = False
                C = A * C
        actual_nits = 0    # To check if algorithm terminates because of threshold,or max iterations reached
        thresh = self.thresh  # Stopping criterion
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            if mask is None:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            else:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                u = mask * u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
                v = mask * v

            err = (u - u1).abs().sum(-1).max()# err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break
        U, V = u, v
        pi = self.exp_M(C, U, V, A=A)
        cost = torch.sum(pi * C, dim=(-2, -1))
        if torch.isnan(cost.sum()):
            print(pi)
            raise
        return cost

    def extract_centerline_2d(self,source,is_gt=False):
        if not is_gt:
            source = torch.sigmoid(source)
            vascular_array = source.detach().cpu().numpy()
        else:
            vascular_array = source.detach().cpu().numpy()
        binary_vascular = (vascular_array > 0.5).astype(np.uint8)
        skeleton = skeletonize(binary_vascular)
        skeleton_tensor = torch.tensor(skeleton, dtype=torch.float32)
        return skeleton_tensor.cuda() * source

    def build_graph_from_torch(self,source,is_gt=False,threshold=0.5,max_num=3000):
        nonzero_indices= []
        nonzero_indices = source.nonzero(as_tuple=False)

        if len(nonzero_indices) > max_num:
            # Get the indices of the top 5000 points based on their values in the source tensor
            sorted_indices = torch.argsort(
                source[nonzero_indices[:, 0], nonzero_indices[:, 1]], descending=True)
            # Use the first 5000 indices to filter the nonzero_indices
            top_indices = sorted_indices[:max_num]
            nonzero_indices = nonzero_indices[top_indices]
        if len(nonzero_indices) == 0:
            # If there are no non-zero points, create a default graph with a single node at (0, 0, 0)
            node_features = torch.tensor([1], dtype=torch.float).view(-1, 1)
            pos = torch.tensor([[0, 0]], dtype=torch.float)
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1]], dtype=torch.float)  # Edge feature (for the single edge)
            degrees = torch.tensor([1], dtype=torch.float)
            data = Data(x=node_features, pos=pos, edge_index=edge_index, edge_attr=edge_attr, degrees=degrees)
            # print(data)
            return data

        node_features = source[nonzero_indices[:, 0], nonzero_indices[:, 1]].view(-1, 1)
        pos = nonzero_indices.clone().detach().view(-1, 2)

        pairwise_distances = torch.cdist(pos.float(), pos.float(), p=2)# Calculate pairwise distances
        edge_mask = pairwise_distances <= 2**0.5# Create mask based on threshold
        edge_indices = edge_mask.nonzero(as_tuple=False)# Get indices of non-zero elements in the mask
        edge_index = torch.stack([edge_indices[:, 0], edge_indices[:, 1]], dim=0)# Create edge_index tensor
        edge_attr = torch.cat([node_features[edge_index[0]], node_features[edge_index[1]]], dim=1) # Extract edge features based on indices
        num_nodes = len(node_features)
        degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        data = Data(x=node_features, pos=pos, edge_index=edge_index, edge_attr=edge_attr,degrees=degrees)
        # data = Data(x=node_features, pos=pos)
        # print(data)
        return data

    def get_cost_matrix(self, source_graph, mask_graph):
        source_positions = source_graph.pos.float()
        mask_positions = mask_graph.pos.float()
        num_nodes_source, num_nodes_mask = len(source_positions), len(mask_positions)

        distances = torch.cdist(source_positions, mask_positions, p=2)
        normalized_cost_matrix = torch.nn.functional.normalize(distances, p=2, dim=1)
        return normalized_cost_matrix

    def get_mask_matrix(self,cost_matrix,num_neighbors = 8):
        num_neighbors = min(num_neighbors, cost_matrix.size(1))
        top_values, top_indices = torch.topk(cost_matrix, num_neighbors, largest=False, dim=1)
        mask_matrix = torch.zeros_like(cost_matrix)
        mask_matrix.scatter_(1, top_indices, 1)# 将每行的前6个最小值的位置设为1
        # print(mask_matrix)

        num_neighbors = min(num_neighbors, cost_matrix.size(0))  # Number of rows
        top_values, top_indices = torch.topk(cost_matrix, num_neighbors, largest=True, dim=0)
        mask_matrix_T = torch.zeros_like(cost_matrix)
        mask_matrix_T.scatter_(0, top_indices, 1)  # 将每列的前6个最大值的位置设为1
        return ((mask_matrix+mask_matrix_T)>0)*1

    def marginal_prob_uniform(self, source_tensor,threshold=0.5):
        neighborhood_kernel = torch.ones(1, 1, 3, 3).to(source_tensor.device)
        coords = torch.nonzero(source_tensor > threshold)
        mask = torch.gt(source_tensor, threshold).float().requires_grad_()  # 这一步有待商榷
        degrees = torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(1), neighborhood_kernel, padding=1,stride=1) * mask
        degree = degrees / degrees.sum()
        degree = degree.squeeze()
        coord = torch.nonzero(degree)
        mu = degree[coord[:, 0], coord[:, 1]]
        gradient_carrying_tensor = source_tensor[coords[:, 0], coords[:, 1]]
        return mu * gradient_carrying_tensor

    def M(self, C, u, v, A=None):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        S = (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        return S

    def exp_M(self, C, u, v, A=None):
        if A is not None:
            if self.sparse:
                a = A.to_dense()
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-a).to(torch.bool),value=0)
            else:
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-A).to(torch.bool),value=0)

            return S
        elif self.mask_matrix is not None:
            return self.mask_matrix * torch.exp(self.M(C, u, v))
        else:
            return torch.exp(self.M(C, u, v))

    def log_sum(self, input_tensor, dim=-1, mask=None):
        s = torch.sum(input_tensor, dim=dim)
        out = torch.log(1e-8 + s)
        if torch.isnan(out.sum()):
            raise
        if mask is not None:
            out = mask * out
        return out

if __name__=='__main__':
    loss=gwDistance()
    for i in range(10):
        n =2
        input_tensor = torch.rand((2,1,n,n))
        zero_input = torch.zeros((2,1,n,n))
        source_tensor = torch.randint(0,3,(2,1,n,n))
        gwd = loss(source_tensor.cuda(),source_tensor.cuda())
        print(gwd)
        print(gwd.requires_grad)