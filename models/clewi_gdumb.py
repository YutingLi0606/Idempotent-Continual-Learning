import torch

from .gdumb import *
from .clewi_mixin import ClewiMixin


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


class ClewiGDumb(ClewiMixin, GDumb):
    NAME = 'clewi_gdumb'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.interpolation_alpha = args.interpolation_alpha
        self.old_model = self.deepcopy_model(backbone)

    def end_task(self, dataset):
        ClewiMixin.end_task(self, dataset)
        GDumb.end_task(self, dataset)
