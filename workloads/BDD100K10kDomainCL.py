import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from clutils.make_experiences import split_dataset
from pathlib import Path
from omegaconf import OmegaConf
import json
from PIL import Image
import os
import sys

# Configuration
sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config
cfg = load_config()

NUM_CLIENTS = cfg.server.num_clients
BATCH_SIZE = cfg.dataset.batch_size

# --- Data --------------------------------------------------------------------
class BDD100K10kDataset(Dataset):
    """Real BDD100K 10k subset dataset loader.
    
    Uses actual images from data/bdd100k_images_10k/ and labels from data/bdd100k_labels/
    for domain incremental learning based on weather and time-of-day attributes.
    """

    def __init__(self, root: str = './data', split: str = 'train', transform=None):
        self.root = Path(root)
        self.split = split  # 'train', 'val', or 'test'
        self.transform = transform
        
        # Paths for images and labels
        self.images_dir = self.root / 'bdd100k_images_10k' / '10k' / split
        self.labels_dir = self.root / 'bdd100k_labels' / '100k' / split
        
        # Load data
        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load image paths, labels, and metadata from the BDD100K dataset."""
        if not self.images_dir.exists():
            print(f"Warning: Images directory {self.images_dir} not found. Using fallback mock data.")
            self._load_mock_data()
            return
            
        if not self.labels_dir.exists():
            print(f"Warning: Labels directory {self.labels_dir} not found. Using fallback mock data.")
            self._load_mock_data()
            return

        # Get all image files
        image_files = list(self.images_dir.glob('*.jpg'))
        
        for img_path in image_files:
            # Get corresponding label file
            img_name = img_path.stem  # filename without extension
            label_path = self.labels_dir / f"{img_name}.json"
            
            if not label_path.exists():
                continue
                
            try:
                # Load label JSON
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                
                # Extract metadata
                attributes = label_data.get('attributes', {})
                weather = attributes.get('weather', 'unknown')
                timeofday = attributes.get('timeofday', 'unknown')
                scene = attributes.get('scene', 'unknown')
                
                # Count cars for simplified classification task
                num_cars = 0
                for frame in label_data.get('frames', []):
                    for obj in frame.get('objects', []):
                        if obj.get('category') == 'car':
                            num_cars += 1
                
                # Cap number of cars at 9 for 10-class classification (0-9)
                label = min(num_cars, 9)
                
                metadata = {
                    'weather': weather,
                    'timeofday': timeofday,
                    'scene': scene,
                    'image_path': str(img_path),
                    'num_objects': len(label_data.get('frames', [{}])[0].get('objects', []))
                }
                
                self.samples.append((str(img_path), label, metadata))
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading label {label_path}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split")

    def _load_mock_data(self):
        """Fallback mock data if real data is not available."""
        print("Using mock data - download real BDD100K data for actual experiments")
        n_total = 1000 if self.split == 'train' else 200
        weather_types = ['clear', 'cloudy', 'rainy', 'overcast']
        tod = ['daytime', 'night']
        
        for i in range(n_total):
            img_path = f"mock_{self.split}_{i}.jpg"
            label = i % 10
            meta = {
                'weather': np.random.choice(weather_types),
                'timeofday': np.random.choice(tod),
                'scene': 'city street',
                'image_path': img_path,
                'num_objects': np.random.randint(1, 10)
            }
            self.samples.append((img_path, label, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, metadata = self.samples[idx]
        
        # Load image
        if img_path.startswith('mock_'):
            # Mock image
            img = torch.randn(3, 224, 224)
        else:
            # Real image
            try:
                with Image.open(img_path) as pil_img:
                    # Convert to RGB if needed
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    # Convert to tensor
                    img = transforms.ToTensor()(pil_img)
                    
                    # Resize to standard size if needed
                    if img.shape[1] != 224 or img.shape[2] != 224:
                        resize_transform = transforms.Resize((224, 224))
                        img = resize_transform(img)
                        
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Fallback to random tensor
                img = torch.randn(3, 224, 224)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, metadata

# Helper datasets to satisfy Avalanche
class TupleDS(Dataset):
    def __init__(self, base_ds, indices=None):
        self.base = base_ds
        self.indices = indices or list(range(len(base_ds)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real = self.indices[idx]
        img, label, _ = self.base[real]
        return img, label

# Domain-specific transforms
DOMAIN_TRANSFORMS = {
    'clear_day': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'clear_night': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3),  # Night images often darker
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'rainy_day': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Rain effect
        transforms.ColorJitter(contrast=0.3, saturation=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'cloudy': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

def _filter_idx(ds, conditions):
    """Filter dataset indices based on metadata conditions."""
    idxs = []
    for i in range(len(ds)):
        _, _, meta = ds[i]
        match = True
        for key, value in conditions.items():
            if isinstance(value, list):
                if meta.get(key) not in value:
                    match = False
                    break
            else:
                if meta.get(key) != value:
                    match = False
                    break
        if match:
            idxs.append(i)
    return idxs

def load_datasets(partition_id: int):
    """Load BDD100K 10k dataset with domain incremental experiences."""
    train_base = BDD100K10kDataset(split='train')
    test_base = BDD100K10kDataset(split='val')  # Use val split for testing
    
    # Simple equal partition across clients
    per_client = len(train_base) // NUM_CLIENTS
    start, end = partition_id * per_client, (partition_id + 1) * per_client
    client_indices = list(range(start, min(end, len(train_base))))
    client_train = Subset(train_base, client_indices)

    # Define domains based on BDD100K attributes
    domains = {
        'clear_day': {'weather': 'clear', 'timeofday': 'daytime'},
        'clear_night': {'weather': 'clear', 'timeofday': 'night'},
        'rainy_day': {'weather': ['rainy', 'partly cloudy'], 'timeofday': 'daytime'},
        'cloudy': {'weather': ['cloudy', 'overcast'], 'timeofday': ['daytime', 'night']},
    }

    # Create training experiences
    train_exps = []
    for name, conditions in domains.items():
        # Get base dataset for filtering
        base_ds = client_train.dataset if isinstance(client_train, Subset) else client_train
        idxs = _filter_idx(base_ds, conditions)
        
        if not idxs:
            print(f"Warning: No samples found for domain {name}")
            continue
            
        # Apply client subset filtering if needed
        if isinstance(client_train, Subset):
            # Filter indices to only include those in client partition
            filtered_idxs = [i for i in idxs if i in client_train.indices]
            if not filtered_idxs:
                continue
            idxs = filtered_idxs
        
        print(f"Domain {name}: {len(idxs)} samples")
        
        # Create dataset for this domain
        exp_ds = TupleDS(base_ds, idxs)
        
        # Apply domain-specific transforms
        transform = DOMAIN_TRANSFORMS.get(name, DOMAIN_TRANSFORMS['clear_day'])
        
        class TransformedDataset(Dataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        transformed_ds = TransformedDataset(exp_ds, transform)
        aval_ds = AvalancheDataset(transformed_ds)
        
        # Split into 2 experiences per domain
        domain_exps = split_dataset(aval_ds, n_experiences=2)
        train_exps.extend(domain_exps)

    # Create test experiences (one per domain)
    test_exps = []
    for name, conditions in domains.items():
        idxs = _filter_idx(test_base, conditions)
        if not idxs:
            continue
            
        exp_ds = TupleDS(test_base, idxs)
        transform = DOMAIN_TRANSFORMS.get(name, DOMAIN_TRANSFORMS['clear_day'])
        
        class TransformedDataset(Dataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        transformed_ds = TransformedDataset(exp_ds, transform)
        aval_ds = AvalancheDataset(transformed_ds)
        test_exps.append(aval_ds)

    # Create benchmark
    benchmark = benchmark_from_datasets(train_datasets=train_exps, test_datasets=test_exps)
    return benchmark


def get_dataloaders(partition_id: int):
    """Get data loaders for BDD100K 10k domain incremental learning."""
    bench = load_datasets(partition_id)
    train_loaders = [DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) 
                    for exp in bench.train_stream]
    test_loaders = [DataLoader(exp.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) 
                   for exp in bench.test_stream]
    return train_loaders, test_loaders 