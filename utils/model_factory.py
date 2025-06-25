# model factory
# automatically determines correct model configuration based on workload and cl strategy


from config_utils import load_config

def get_workload_info(workload_name, cl_strategy, num_experiences):
    """get dataset and model information for a workload"""
    
    workload_configs = {
        # standard datasets - same classes across all tasks
        'cifar10': {
            'total_classes': 10,
            'classes_per_task': 10,  # all classes available in each task
            'task_type': 'domain',
            'input_channels': 3,  # RGB images
            'input_size': 32  # CIFAR is 32x32
        },
        'cifar100': {
            'total_classes': 100, 
            'classes_per_task': 100,
            'task_type': 'domain',
            'input_channels': 3,  # RGB images
            'input_size': 32  # CIFAR is 32x32
        },
        'permuted_mnist': {
            'total_classes': 10,
            'classes_per_task': 10,  # same classes, different permutations
            'task_type': 'domain',
            'input_channels': 1,  # grayscale images
            'input_size': 28  # MNIST is 28x28
        },
        
        # class-incremental datasets - classes split across tasks
        'split_cifar10': {
            'total_classes': 10,
            'classes_per_task': lambda exp: 10 // exp,  # split classes
            'task_type': 'class_incremental',
            'input_channels': 3,  # RGB images
            'input_size': 32  # CIFAR is 32x32
        },
        'icifar100': {
            'total_classes': 100,
            'classes_per_task': lambda exp: 100 // exp,  # split classes
            'task_type': 'class_incremental',
            'input_channels': 3,  # RGB images
            'input_size': 32  # CIFAR is 32x32
        },
        
        # real-world datasets
        'bdd100k': {
            'total_classes': 10,  # car count classification
            'classes_per_task': 10,
            'task_type': 'domain',
            'input_channels': 3,  # RGB images
            'input_size': 224  # typical for real-world datasets
        },
        'bdd100k_v2': {
            'total_classes': 10,
            'classes_per_task': 10, 
            'task_type': 'domain',
            'input_channels': 3,  # RGB images
            'input_size': 224
        },
        'bdd100k_10k': {
            'total_classes': 10,
            'classes_per_task': 10,
            'task_type': 'domain',
            'input_channels': 3,  # RGB images
            'input_size': 224
        },
        'kitti': {
            'total_classes': 10,
            'classes_per_task': 10,
            'task_type': 'domain',
            'input_channels': 3,  # RGB images
            'input_size': 224
        },
        'kitti_v2': {
            'total_classes': 10,
            'classes_per_task': 10,
            'task_type': 'domain',
            'input_channels': 3,  # RGB images
            'input_size': 224
        }
    }
    
    if workload_name not in workload_configs:
        print(f"warning: unknown workload {workload_name}, using default config")
        return {
            'total_classes': 10,
            'classes_per_task': 10,
            'task_type': 'domain',
            'input_channels': 3,  # default to RGB
            'input_size': 32  # default to CIFAR size
        }
    
    config = workload_configs[workload_name].copy()
    
    # calculate classes per task for split datasets
    if callable(config['classes_per_task']):
        config['classes_per_task'] = config['classes_per_task'](num_experiences)
    
    return config

def get_model_classes(cfg):
    """intelligently determine number of classes for model"""
    
    # get workload info
    workload_info = get_workload_info(
        cfg.dataset.workload,
        cfg.cl.strategy, 
        cfg.cl.num_experiences
    )
    
    # for class-incremental tasks, use classes per task
    if workload_info['task_type'] == 'class_incremental':
        model_classes = workload_info['classes_per_task']
        print(f"class-incremental workload: {model_classes} classes per task")
    else:
        # for domain tasks, use total classes
        model_classes = workload_info['total_classes']
        print(f"domain workload: {model_classes} total classes")
    
    # allow manual override in model config
    if hasattr(cfg.model, 'num_classes') and cfg.model.num_classes != 10:
        print(f"manual override: using {cfg.model.num_classes} classes from model config")
        model_classes = cfg.model.num_classes
    
    return model_classes, workload_info

def create_model(cfg):
    
    model_classes, workload_info = get_model_classes(cfg)
    input_channels = workload_info.get('input_channels', 3)  # default to RGB
    input_size = workload_info.get('input_size', 32)  # default to CIFAR size
    
    print(f"creating {cfg.model.name} with {model_classes} output classes, {input_channels} input channels, {input_size}x{input_size} input size")
    
    if cfg.model.name == "resnet":
        from models.ResNet import ResNet
        return ResNet(num_classes=model_classes)
    
    elif cfg.model.name == "simple_cnn":
        from models.SimpleCNN import create_simple_cnn
        input_size = workload_info.get('input_size', 32)  # default to CIFAR size
        return create_simple_cnn(num_classes=model_classes, in_channels=input_channels, input_size=input_size)
    
    elif cfg.model.name == "mobilenet":
        from models.MobileNet import create_mobilenet
        version = getattr(cfg.model, 'version', 'v2')
        pretrained = getattr(cfg.model, 'pretrained', False)
        return create_mobilenet(
            num_classes=model_classes, 
            pretrained=pretrained, 
            version=version
        )
    
    else:
        raise ValueError(f"unknown model: {cfg.model.name}")

# backward compatibility aliases
def validate_config(cfg):
    """validate that workload and model configuration are compatible"""
    model_classes, workload_info = get_model_classes(cfg)
    
    warnings = []
    
    # check if class-incremental with wrong strategy
    if (workload_info['task_type'] == 'class_incremental' and 
        cfg.cl.strategy in ['domain']):
        warnings.append(f"workload '{cfg.dataset.workload}' is class-incremental but using domain strategy")
    
    # check if too many experiences for split datasets  
    if (workload_info['task_type'] == 'class_incremental' and
        cfg.cl.num_experiences > workload_info['total_classes']):
        warnings.append(f"too many experiences ({cfg.cl.num_experiences}) for {workload_info['total_classes']} classes")
    
    # check experience replay buffer size
    if (hasattr(cfg.cl, 'replay_mem_size') and cfg.cl.replay_mem_size > 1000):
        warnings.append(f"large replay buffer ({cfg.cl.replay_mem_size}) may cause memory issues")
    
    if warnings:
        print(" configuration warnings... :")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print("configuration validated successfully")
    
    return warnings

def get_model(cfg=None):
    """backward compatibility wrapper"""
    if cfg is None:
        cfg = load_config()
    return create_model(cfg) 