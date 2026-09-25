import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tasks import DelayedResponse, ReactionTime, CategoryDecision, DelayedMatchToSample, ContextIntegration, GoNogo
from tasks.dataset import HierarchicalTasksDataset, MultiTaskDataset, collate_fn
from torch.utils.data import DataLoader
from hierachical_model_task.bptt import BPTT
from hierachical_model_task.utils import HiearchicalModelConfig

duration_params = {
    'context': (5, 10),      # Shorter
    'stimulus': (10, 15),    # Shorter, less variable
    'delay': (10, 15),       # Shorter delays
    'response': (5, 10)
}

TASKS_MAP = {
    'DelayPro': DelayedResponse(duration_params, mode='pro'),
    'DelayAnti': DelayedResponse(duration_params, mode='anti'),
    'ReactPro': ReactionTime(duration_params, mode='pro'),
    'ReactAnti': ReactionTime(duration_params, mode='anti'),
    'CatPro': CategoryDecision(duration_params, mode='pro'),
    'CatAnti': CategoryDecision(duration_params, mode='anti'),
    'Match2Sample': DelayedMatchToSample(duration_params, mode='match'),
    'NonMatch2Sample': DelayedMatchToSample(duration_params, mode='nonmatch'),
    'CtxIntMod1': ContextIntegration(duration_params, relevant_modality=1), 
    'CtxIntMod2': ContextIntegration(duration_params, relevant_modality=2),
    'GoNogo': GoNogo(duration_params),
}


task_names = list(TASKS_MAP.keys())


def _get_device(args: HiearchicalModelConfig):
    args.device = 'cpu'
    if args.use_gpu:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        try:
            args.device = args.device + ':' + str(args.device_id)
        except AttributeError:
            print("Must specify device id when using cuda. This should be handeled by the task distribution.")
    print(f'Using device: {args.device}')
    return args

def _handle_defaults(args: HiearchicalModelConfig):
    # set learning rates
    if args.individual_learning_rate is not None:
        args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    else:
        args.learning_rate = (args.learning_rate, args.learning_rate)
    # set teacher forcing alpha
    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start
    return args

def main(tasks = None):
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    if tasks is None:
        tasks = [TASKS_MAP[name] for name in TASKS_MAP]
    print(f"Training on {len(tasks)} tasks: {', '.join(task_names)}")
    
    # Set console args programmatically
    args = HiearchicalModelConfig(tasks=tasks)
    args = _get_device(args)
    args = _handle_defaults(args)

    train_dataset = HierarchicalTasksDataset(tasks, n_trials=len(tasks)*200)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataset = HierarchicalTasksDataset(tasks, n_trials=len(tasks)*50, task_indices=[i for i in range(len(tasks)) for _ in range(50)], fixed=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=True,
        collate_fn=collate_fn
 )

    training_alg = BPTT(None, args)
    loss_history, test_loss_history, train_task_accuracies, test_task_accuracies = training_alg.train(train_loader, test_loader)


if __name__ == '__main__':
    main()