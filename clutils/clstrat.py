import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import (
    forgetting_metrics, 
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics, 
    confusion_matrix_metrics,
    disk_usage_metrics,
    )

from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, EWCPlugin
from avalanche.training.templates import SupervisedTemplate

from avalanche.training.supervised import Naive, EWC, Replay, ICaRL

from omegaconf import OmegaConf
from pathlib import Path

# Setting up Config Path
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
cfg = OmegaConf.load(config_path)

strategy = cfg.cl.strategy 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_cl_strat(net):
    # log to text file
    text_logger = TextLogger(open('logs/avalog.txt', 'a'))
    # print to stdout
    interactive_logger = InteractiveLogger()

    # Define Avalanche Eval Plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=10, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger]
    )

    if strategy == "naive":
        cl_strategy = Naive(
            net,
            Adam(net.parameters()),
            CrossEntropyLoss(),
            train_mb_size=cfg.dataset.batch_size,
            train_epochs=cfg.client.epochs,
            eval_mb_size=cfg.dataset.batch_size,
            evaluator=eval_plugin,
            device=DEVICE
            )
    if strategy == "ewc":
        cl_strategy = EWC(
            net,
            Adam(net.parameters()),
            CrossEntropyLoss(),
            ewc_lambda=cfg.cl.ewc.lmb,
            train_mb_size=cfg.dataset.batch_size,
            eval_mb_size=cfg.dataset.batch_size,
            evaluator=eval_plugin,
            device=DEVICE
        )
    if strategy == "replay":
        cl_strategy = Replay(
            net, 
            Adam(net.parameters()),
            CrossEntropyLoss(),
            mem_size = cfg.cl.replay.mem_size,
            train_mb_size=cfg.dataset.batch_size,
            eval_mb_size=cfg.datasaet.batch_size,
            evaluator=eval_plugin,
            device=DEVICE
        )
    if strategy == "ewc-replay":
        replay = ReplayPlugin(mem_size=cfg.cl.replay.mem_size)
        ewc = EWCPlugin(ewc_lambda=cfg.cl.ewc.lmb)
        cl_strategy = SupervisedTemplate(
            net,
            Adam(net.parameters()),
            CrossEntropyLoss(),
            plugins=[replay, ewc],
            train_mb_size=cfg.dataset.batch_size,
            test_mb_size=cfg.dataset.batch_size,
            evaluator=eval_plugin,
            device=DEVICE
        )
        

    return cl_strategy, eval_plugin
