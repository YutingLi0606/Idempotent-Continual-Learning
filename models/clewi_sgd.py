import torch

from .sgd import *
from .clewi_mixin import ClewiMixin
from utils.args import add_rehearsal_args
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--maxlr', type=float, default=5e-2,
                        help='Penalty weight.')
    parser.add_argument('--minlr', type=float, default=5e-4,
                        help='Penalty weight.')
    parser.add_argument('--fitting_epochs', type=int, default=256,
                        help='Penalty weight.')
    parser.add_argument('--cutmix_alpha', type=float, default=None,
                        help='Penalty weight.')
    add_experiment_args(parser)

    parser.add_argument('--interpolation_alpha', type=float, default=0.5, help='interpolation alpha')
    parser.add_argument('--debug_interpolation', action='store_true')
    return parser


class ClewiSgd(ClewiMixin, Sgd):
    NAME = 'clewi_sgd'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.interpolation_alpha = args.interpolation_alpha
        self.old_model = self.deepcopy_model(backbone)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        ClewiMixin.end_task(self, dataset)
        # Sgd.end_task(self, dataset)
