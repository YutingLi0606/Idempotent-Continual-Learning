import torch

from .xder import *
from .clewi_mixin import ClewiMixin


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')

    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--lambd', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--m', type=float, default=0.3)

    parser.add_argument('--simclr_temp', type=float, default=5)
    parser.add_argument('--simclr_batch_size', type=int, default=64)
    parser.add_argument('--simclr_num_aug', type=int, default=2)
    parser.add_argument('--interpolation_alpha', type=float, default=0.5,
                        help='interpolation alpha')
    parser.add_argument('--debug_interpolation', action='store_true')
    return parser


class ClewiXDer(ClewiMixin, XDer):
    NAME = 'clewi_xder'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.interpolation_alpha = args.interpolation_alpha
        self.old_model = self.deepcopy_model(backbone)

    def end_task(self, dataset):
        ClewiMixin.end_task(self, dataset)
        XDer.end_task(self, dataset)
