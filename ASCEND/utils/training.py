# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv
import math
import sys
from argparse import Namespace
from typing import Tuple
import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple, List
from utils.loggers import *
from utils.mlflow_logger import MLFlowLogger
from utils.status import ProgressBar
import torch.nn.functional as F
import utils.metrics


def evaluate_ece(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[List[float], List[float], float, float, float]:
    """
    Evaluates accuracy and computes AURC, FPR95, AUROC for each loader, then averages.
    :param model: model to evaluate
    :param dataset: continual dataset
    :return: class-il acc, task-il acc, mean AURC, mean FPR95, mean AUROC
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    ece_list = []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue

        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[]}
        
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)

                if 'class-il' not in model.COMPATIBILITY:
                    output = model(inputs, k)
                else:
                    output = model(inputs)
                softmax = F.softmax(output, dim=1)
                _, pred_cls = softmax.max(1)

                val_log['correct'].append(pred_cls.cpu().eq(labels.cpu().data.view_as(pred_cls)).numpy())
                val_log['softmax'].append(softmax.cpu().data.numpy())
                val_log['logit'].append(output.cpu().data.numpy())
                val_log['target'].append(labels.cpu().data.numpy())
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                if dataset.SETTING == 'class-il':
                    mask_classes(output, dataset, k)
                    _, pred_masked = torch.max(output.data, 1)
                    correct_mask_classes += pred_masked.eq(labels).sum().item()
        for key in val_log : 
            val_log[key] = np.concatenate(val_log[key])
        acc = 100. * val_log['correct'].mean()
        ece = utils.metrics.calc_ece(val_log['softmax'], val_log['target'], bins=15)
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        ece_list.append(ece*100)
    model.net.train(status)

    # averagge
    mean_ece = np.mean(ece_list)
    print('evaluation acc:', accs)
    print(f'ECE per loader: {ece_list}')
    print(f'Mean ECE: {mean_ece:.4f}')

    return accs, accs_mask_classes
def evaluate_eceid(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[List[float], List[float], float, float, float]:
    """
    Evaluates accuracy and computes AURC, FPR95, AUROC for each loader, then averages.
    :param model: model to evaluate
    :param dataset: continual dataset
    :return: class-il acc, task-il acc, mean AURC, mean FPR95, mean AUROC
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    ece_list = []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[]}
        
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                batch_size, _, H, W = inputs.shape
                if 'class-il' not in model.COMPATIBILITY:
                    output = model(inputs, k)
                else:
                    y_0 = torch.ones(batch_size,  dataset.N_CLASSES).to(inputs.device) /dataset.N_CLASSES
                    z = model.net.f1(inputs)
                    output, z1 = model.net.f2(z, y_0)
                softmax = F.softmax(output, dim=1)
                _, pred_cls = softmax.max(1)

                val_log['correct'].append(pred_cls.cpu().eq(labels.cpu().data.view_as(pred_cls)).numpy())
                val_log['softmax'].append(softmax.cpu().data.numpy())
                val_log['logit'].append(output.cpu().data.numpy())
                val_log['target'].append(labels.cpu().data.numpy())
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                if dataset.SETTING == 'class-il':
                    mask_classes(output, dataset, k)
                    _, pred_masked = torch.max(output.data, 1)
                    correct_mask_classes += pred_masked.eq(labels).sum().item()
        for key in val_log : 
            val_log[key] = np.concatenate(val_log[key])
        acc = 100. * val_log['correct'].mean()
        ece = utils.metrics.calc_ece(val_log['softmax'], val_log['target'], bins=15)
        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        ece_list.append(ece*100)

    model.net.train(status)

    # average
    mean_ece = np.mean(ece_list)
    # 
    print('evaluation acc:', accs)
    print(f'ECE per loader: {ece_list}')
    print(f'Mean ECE: {mean_ece:.4f}')

    return accs, accs_mask_classes



def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    print('evaluation acc:')
    print(accs)
    return accs, accs_mask_classes


def evaluateid(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
           
                batch_size, _, H, W = inputs.shape
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    y_0 = torch.ones(batch_size,  dataset.N_CLASSES).to(inputs.device) /dataset.N_CLASSES
                    z = model.net.f1(inputs)
                    outputs, z1 = model.net.f2(z, y_0)



                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    print('evaluation acc:')
    print(accs)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    model.net.to(model.device)
    results, results_mask_classes = [], []
    

    if not args.disable_log and not args.debug:
        logger = MLFlowLogger(dataset.SETTING, dataset.NAME, model.NAME,
                              experiment_name=args.experiment_name, parent_run_id=args.parent_run_id, run_name=args.run_name)
        logger.log_args(args.__dict__)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics and not args.debug:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            if model.NAME =='ider':
                random_results_class, random_results_task = evaluateid(model, dataset_copy)
            else:
                random_results_class, random_results_task = evaluate(model, dataset_copy)

    if os.path.exists('old_model.pt'):
        os.remove('old_model.pt')
    if os.path.exists('net.pt'):
        os.remove('net.pt')

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics and not args.debug:
            if model.NAME =='ider':
                accs = evaluateid(model, dataset, last=True)
            else:
                accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):          
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()
            if hasattr(model, 'end_epoch'):
                model.end_epoch(dataset)


        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if model.NAME =='ider':
            accs = evaluateid(model, dataset)
        else:
            accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        if model.NAME =='ider':
            eces= evaluate_eceid(model, dataset)
        else: 
            eces= evaluate_ece(model, dataset)
        
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log and not args.debug:
            logger.log(mean_acc)
            logger.log_fullacc(accs)
  
    if not args.disable_log and not args.ignore_other_metrics and not args.debug:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if args.savecheckpoint:
        dataset_name = args.dataset if hasattr(args, 'dataset') and args.dataset else 'unknown_dataset'
        buffer_tag = f"buffer_{args.buffer_size}" if hasattr(args, 'buffer_size') and args.buffer_size is not None else "buffer_none"
        save_dir = os.path.join("./experiments", dataset_name, buffer_tag)
        os.makedirs(save_dir, exist_ok=True)

        model_filename = f"{model.NAME}_seed_{args.seed}.pth"
        model_path = os.path.join(save_dir, model_filename)

        torch.save(model.net.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
