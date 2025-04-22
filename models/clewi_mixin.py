import torch
import torch.utils.data
import copy

from utils.buffer import Buffer
from models.utils.weight_interpolation import *

from .clewi import evaluate


class ClewiMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.first_task = True

    def end_task(self, dataset):
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
            return

        buffer_dataloder = self.get_buffer_dataloder()
        torch.save(self.old_model, 'old_model.pt')
        torch.save(self.net, 'net.pt')

        buffer_dataloder = self.get_buffer_dataloder()
        if self.args.debug_interpolation:
            self.interpolation_plot(dataset, buffer_dataloder)

        self.old_model = interpolate(self.net, self.old_model, buffer_dataloder, self.device, alpha=self.interpolation_alpha)
        self.net = self.deepcopy_model(self.old_model)
        self.opt = self.opt.__class__(self.net.parameters(), **self.opt.defaults)
        self.opt.zero_grad()

    def get_buffer_dataloder(self):
        data = self.buffer.get_data(len(self.buffer), transform=self.transform)
        buf_inputs = data[0]
        buf_labels = data[1]

        buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=32, num_workers=0)
        return buffer_dataloder

    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy

    def interpolation_plot(self, dataset, buffer_dataloader):
        alpha_grid = np.arange(0, 1.001, 0.02)
        interpolation_accs = []
        for alpha in alpha_grid:
            net = self.deepcopy_model(self.net)
            old = self.deepcopy_model(self.old_model)
            new_model = interpolate(net, old, buffer_dataloader, self.device, alpha=alpha)
            acc = evaluate(new_model, dataset, self.device)
            interpolation_accs.append(acc)
        print('interpolation accuracies:')
        print(interpolation_accs)
        print('old model accs:')
        accs = evaluate(self.old_model, dataset, self.device)
        print(accs)
        print('new model accs:')
        accs = evaluate(self.net, dataset, self.device)
        print(accs)
