import torch
import gc  # For garbage collection
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import (
    forgetting_metrics, 
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics, 
    disk_usage_metrics,
    )
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

# import all continual learning strategies
from avalanche.training.supervised import Naive, EWC, Replay
from avalanche.training.plugins import ReplayPlugin, EWCPlugin

from algorithms.plora import PLoRAStrategy
from algorithms.fedcprompt import FedCPromptStrategy
from algorithms.fedet import FedETStrategy
from algorithms.fedgem import FedGEMStrategy
from algorithms.fedma import FedMAStrategy
from algorithms.fedproto import FedProtoStrategy
from algorithms.fedrcil import FedRCILStrategy
from algorithms.fedrep import FedRepStrategy
from algorithms.fedweit import FedWeITStrategy
from algorithms.sacfl import SacFLStrategy, SacFLEncoderDecoder
from algorithms.stamp import STAMPStrategy

from omegaconf import OmegaConf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config

cfg = load_config()

# Respect configuration: only use GPU if num_gpus > 0.0 AND CUDA is available
if cfg.client.num_gpus > 0.0 and torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def make_cl_strat(net):
    """create continual learning strategy based on configuration"""
    
    # Clear any existing CUDA cache before creating strategy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # setup logging
    text_logger = TextLogger(open('logs/avalog.txt', 'a'))
    interactive_logger = InteractiveLogger()

    # Only compute standard metrics (no confusion matrix)
    metrics = [
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    ]

    eval_plugin = EvaluationPlugin(
        *metrics,
        loggers=[interactive_logger, text_logger]
    )

    # get strategy configuration
    strategy_name = getattr(cfg.cl, 'strategy', 'naive')
    
    # common strategy parameters
    common_params = {
        'model': net,
        'optimizer': Adam(net.parameters(), lr=cfg.training.learning_rate),
        'criterion': CrossEntropyLoss(),
        'train_mb_size': cfg.dataset.batch_size,
        'train_epochs': cfg.client.epochs,
        'eval_mb_size': cfg.dataset.batch_size,
        'evaluator': eval_plugin,
        'device': DEVICE
    }
    
    # SOTA strategies
    if strategy_name == 'plora':
        plora_config = getattr(cfg, 'plora', {})
        cl_strategy = PLoRAStrategy(net, plora_config, num_clients=cfg.server.num_clients)
        return cl_strategy, None
    elif strategy_name == 'fedcprompt':
        fedcprompt_config = getattr(cfg, 'fedcprompt', {})
        cl_strategy = FedCPromptStrategy(net, fedcprompt_config, num_clients=cfg.server.num_clients, num_tasks=cfg.cl.num_experiences)
        return cl_strategy, None
    elif strategy_name == 'fedet':
        fedet_config = getattr(cfg, 'fedet', {})
        cl_strategy = FedETStrategy(net, fedet_config, num_clients=cfg.server.num_clients, num_tasks=cfg.cl.num_experiences)
        return cl_strategy, None
    elif strategy_name == 'fedgem':
        gem_config = getattr(cfg, 'gem', {})
        cl_strategy = FedGEMStrategy(net, gem_config, num_clients=cfg.server.num_clients, memory_size=gem_config.get('memory_size', 200))
        return cl_strategy, None
    elif strategy_name == 'fedma':
        ma_config = getattr(cfg, 'fedma', {})
        cl_strategy = FedMAStrategy(net, ma_config, num_clients=cfg.server.num_clients)
        return cl_strategy, None
    elif strategy_name == 'fedproto':
        proto_config = getattr(cfg, 'fedproto', {})
        cl_strategy = FedProtoStrategy(net, proto_config, num_clients=cfg.server.num_clients)
        return cl_strategy, None
    elif strategy_name == 'fedrcil':
        rc_config = getattr(cfg, 'fedrcil', {})
        cl_strategy = FedRCILStrategy(net, rc_config, num_clients=cfg.server.num_clients, buffer_size=rc_config.get('buffer_size', 200))
        return cl_strategy, None
    elif strategy_name == 'fedrep':
        rep_config = getattr(cfg, 'fedrep', {})
        rep_layer_names = rep_config.get('rep_layer_names', [])
        cl_strategy = FedRepStrategy(net, rep_layer_names, device=str(DEVICE))
        return cl_strategy, None
    elif strategy_name == 'fedweit':
        weit_config = getattr(cfg, 'fedweit', {})
        cl_strategy = FedWeITStrategy(net, sparsity=weit_config.get('sparsity', 0.5), num_clients=cfg.server.num_clients, l1_lambda=weit_config.get('l1_lambda', 0.1), l2_lambda=weit_config.get('l2_lambda', 100.0), device=str(DEVICE))
        return cl_strategy, None
    elif strategy_name == 'sacfl':
        sacfl_config = getattr(cfg, 'sacfl', {})
        # For SacFL, model must be encoder-decoder
        cl_strategy = SacFLStrategy(net, sacfl_config, num_clients=cfg.server.num_clients)
        return cl_strategy, None
    elif strategy_name == 'stamp':
        stamp_config = getattr(cfg, 'stamp', {})
        cl_strategy = STAMPStrategy(net, stamp_config, num_clients=cfg.server.num_clients)
        return cl_strategy, None

    # Avalanche strategies (default)
    if strategy_name == 'naive' or strategy_name == 'domain':
        cl_strategy = Naive(**common_params)
    elif strategy_name == 'ewc':
        ewc_lambda = getattr(cfg.cl, 'ewc_lambda', 0.4)  # default from literature
        decay_factor = getattr(cfg.cl, 'ewc_decay_factor', None)
        keep_importance_data = getattr(cfg.cl, 'ewc_keep_importance_data', False)
        cl_strategy = EWC(
            ewc_lambda=ewc_lambda,
            decay_factor=decay_factor,
            keep_importance_data=keep_importance_data,
            **common_params
        )
    elif strategy_name == 'replay':
        # experience replay
        mem_size = getattr(cfg.cl, 'replay_mem_size', 200)  # buffer size
        cl_strategy = Replay(
            mem_size=mem_size,
            **common_params
        )
        
    elif strategy_name == 'hybrid':
        # hybrid: ewc + replay
        ewc_lambda = getattr(cfg.cl, 'ewc_lambda', 0.4)
        mem_size = getattr(cfg.cl, 'replay_mem_size', 200)
        
        # create plugins for hybrid approach
        plugins = [
            EWCPlugin(ewc_lambda=ewc_lambda),
            ReplayPlugin(mem_size=mem_size)
        ]
        cl_strategy = Naive(
            plugins=plugins,
            **common_params
        )
    else:
        print(f"warning: unknown strategy '{strategy_name}', falling back to naive")
        cl_strategy = Naive(**common_params)
    print(f"created continual learning strategy: {strategy_name}")
    if hasattr(cfg.cl, 'ewc_lambda'):
        print(f"  ewc lambda: {cfg.cl.ewc_lambda}")
    if hasattr(cfg.cl, 'replay_mem_size'):
        print(f"  replay buffer size: {cfg.cl.replay_mem_size}")
    return cl_strategy, eval_plugin
