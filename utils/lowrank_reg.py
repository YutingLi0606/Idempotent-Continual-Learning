import torch
import torch.nn as nn
import torch.nn.functional as F



class LowRankReg(nn.Module):
    def __init__(self, dim, lr=1e-4, rank=None):
        super(LowRankReg, self).__init__()
        self.dim = dim
        if rank is None: rank = dim
        self.rank = rank
        self.lam1 = 1.0
        self.lam2 = 1.0
        self.lr = lr

        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.bias = nn.Parameter(torch.Tensor(1, dim))
        self.reset_parameters()
        self.reset_optimizer()

        self.step_count = 0

    def reset_parameters(self):
        # nn.init.eye_(self.weight)
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def set_rank(self, rank):
        self.rank = rank

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        '''
        Compute the project tensor
        A: [batch, dim]
        '''
        Z = torch.mm(A + self.bias, self.weight.T)
        return Z

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def forward_and_loss(self, A: torch.Tensor) -> torch.Tensor:
        '''
        Compute the project tensor and the loss
        A: [batch, dim]
        '''
        n = A.size(0)

        # L2 normalize the input
        A = F.normalize(A, p=2, dim=1)

        Z = self(A)
        loss_recon = torch.frobenius_norm(Z - (A + self.bias)).pow(2) / n

        loss = self.lam1 * loss_recon
        
        # loss_norm = (1-torch.linalg.vector_norm(A, ord=2, dim=1)).abs().mean()
        # loss = self.lam1 * loss_recon + self.lam2 * loss_norm

        # # logging for debugging
        # wandb.log({
        #     'lr/loss_recon': self.lam1 * loss_recon.item(),
        #     'lr/loss_norm': self.lam2 * loss_norm.item(),
        # })

        return Z, loss

    def SVP(self, rank:int=None) -> None:
        '''
        Singular value projection of the weight matrix
        '''
        if rank is None: rank = self.rank

        if rank == self.weight.size(0): return

        u, s, v = torch.svd(self.weight)
        s[rank:] = 0
        weight = torch.mm(u, torch.diag(s))
        weight = torch.mm(weight, v.T)
        self.weight = nn.Parameter(weight)
        self.reset_optimizer()
