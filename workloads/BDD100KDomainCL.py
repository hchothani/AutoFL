import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import flwr
from flwr_datasets import FederatedDataset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
from clutils.make_experiences import split_dataset

NUM_CLIENTS = 10
BATCH_SIZE = 32

# Define base transforms
BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def filter_by_attributes(dataset, attribute_name, attribute_value):
    """Filter dataset by specific attribute value"""
    filtered_indices = []
    for idx, sample in enumerate(dataset):
        if sample['attributes'][attribute_name] == attribute_value:
            filtered_indices.append(idx)
    return filtered_indices

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="bdd100k", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Define domains based on weather and time of day
    domains = {
        'clear_day': {'weather': 'clear', 'timeofday': 'daytime'},
        'clear_night': {'weather': 'clear', 'timeofday': 'night'},
        'cloudy_day': {'weather': 'cloudy', 'timeofday': 'daytime'},
        'cloudy_night': {'weather': 'cloudy', 'timeofday': 'night'},
        'rainy_day': {'weather': 'rainy', 'timeofday': 'daytime'},
        'rainy_night': {'weather': 'rainy', 'timeofday': 'night'},
        'snowy_day': {'weather': 'snowy', 'timeofday': 'daytime'},
        'snowy_night': {'weather': 'snowy', 'timeofday': 'night'}
    }
    
    # Create experiences for each domain
    train_experiences = []
    for domain_name, attributes in domains.items():
        # Filter data for this domain
        weather_indices = filter_by_attributes(
            partition_train_test["train"], 
            'weather', 
            attributes['weather']
        )
        time_indices = filter_by_attributes(
            partition_train_test["train"], 
            'timeofday', 
            attributes['timeofday']
        )
        domain_indices = list(set(weather_indices) & set(time_indices))
        
        if len(domain_indices) > 0:
            # Create domain dataset
            domain_data = partition_train_test["train"].select(domain_indices)
            domain_data = domain_data.with_transform(
                lambda x: {'img': BASE_TRANSFORM(x['img']), 'label': x['label']}
            )
            
            # Split into experiences
            domain_experiences = split_dataset(domain_data, n_experiences=2)
            train_experiences.extend(domain_experiences)
    
    # Create test experiences
    test_experiences = []
    testset = fds.load_split("test")
    for domain_name, attributes in domains.items():
        # Filter test data for this domain
        weather_indices = filter_by_attributes(testset, 'weather', attributes['weather'])
        time_indices = filter_by_attributes(testset, 'timeofday', attributes['timeofday'])
        domain_indices = list(set(weather_indices) & set(time_indices))
        
        if len(domain_indices) > 0:
            domain_test = testset.select(domain_indices)
            domain_test = domain_test.with_transform(
                lambda x: {'img': BASE_TRANSFORM(x['img']), 'label': x['label']}
            )
            test_experiences.append(domain_test)
    
    # Create benchmark
    benchmark = benchmark_from_datasets(
        train_datasets=train_experiences,
        test_datasets=test_experiences
    )
    
    return benchmark

def get_dataloaders(partition_id: int):
    benchmark = load_datasets(partition_id)
    
    # Create dataloaders for each experience
    train_loaders = []
    for exp in benchmark.train_stream:
        train_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=True)
        )
    
    test_loaders = []
    for exp in benchmark.test_stream:
        test_loaders.append(
            DataLoader(exp.dataset, batch_size=BATCH_SIZE)
        )
    
    return train_loaders, test_loaders 