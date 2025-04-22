import torch

from .bic import *
from .clewi_mixin import ClewiMixin


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='A bag of tricks for Continual learning.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--bic_epochs', type=int, default=250,
                        help='bias injector.')
    parser.add_argument('--temp', type=float, default=2.,
                        help='softmax temperature')
    parser.add_argument('--valset_split', type=float, default=0.1,
                        help='bias injector.')
    parser.add_argument('--multi_bic', type=int, default=0)
    parser.add_argument('--wd_reg', type=float, default=None,
                        help='bias injector.')
    parser.add_argument('--distill_after_bic', type=int, default=1)
    parser.add_argument('--interpolation_alpha', type=float, default=0.5,
                        help='interpolation alpha')
    parser.add_argument('--debug_interpolation', action='store_true')

    return parser


class ClewiBiC(ClewiMixin, BiC):
    NAME = 'clewi_bic'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.interpolation_alpha = args.interpolation_alpha
        self.old_model = self.deepcopy_model(backbone)

    def begin_task(self, dataset):
        if self.task > 0:
            self.net.train()
            self.lamda = 1 / (self.task + 1)

            icarl_replay(self, dataset, val_set_split=self.args.valset_split)

        if hasattr(self, 'corr_factors'):
            del self.corr_factors

    def end_task(self, dataset):
        self.evaluate_corr(dataset)
        self.old_net = deepcopy(self.net.eval())
        if hasattr(self, 'corr_factors'):
            self.old_corr = deepcopy(self.corr_factors)
        self.net.train()

        self.build_buffer(dataset, self.task+1)
        ClewiMixin.end_task(self, dataset)
        self.evaluate_corr(dataset)

        self.task += 1

    def evaluate_corr(self, dataset):
        if self.task > 0:
            self.net.eval()

            from utils.training import evaluate
            print("EVAL PRE", evaluate(self, dataset))

            self.evaluate_bias('pre')

            corr_factors = torch.tensor([0., 1.], device=self.device, requires_grad=True)
            self.biasopt = Adam([corr_factors], lr=0.001)

            for l in range(self.args.bic_epochs):
                for inputs, labels, _ in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.biasopt.zero_grad()
                    with torch.no_grad():
                        out = self.forward(inputs)

                    start_last_task = (self.task) * self.cpt
                    end_last_task = (self.task + 1) * self.cpt
                    tout = out + 0
                    tout[:, start_last_task:end_last_task] *= corr_factors[1].repeat_interleave(end_last_task - start_last_task)
                    tout[:, start_last_task:end_last_task] += corr_factors[0].repeat_interleave(end_last_task - start_last_task)

                    loss_bic = self.loss(tout[:, :end_last_task], labels)
                    loss_bic.backward()
                    self.biasopt.step()

            self.corr_factors = corr_factors
            print(self.corr_factors, file=sys.stderr)

            self.evaluate_bias('post')

            self.net.train()
