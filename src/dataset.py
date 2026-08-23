from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Any, Dict, List, Tuple


def get_dataloaders(config_yaml: Dict[str, Dict[Any, Any]]) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
	
	"""Create stratified train, validation, and test data loaders.

	The dataset root must contain one subdirectory per class, as expected by
	``torchvision.datasets.ImageFolder``. Ratios can be supplied in the
	``data`` configuration section; they default to 80/10/10.
	"""
	data_config = config_yaml["data"]
	dataset_path = Path(data_config["raw_dataset"])
	image_size = data_config.get("img_size", 224)
	batch_size = data_config.get("batch_size", 32)
	num_workers = data_config.get("num_workers", 0)
	train_ratio = data_config.get("train_ratio", 0.80)
	validation_ratio = data_config.get("val_ratio", 0.10)
	test_ratio = data_config.get("test_ratio", 0.10)
	seed = config_yaml.get("training", {}).get("seed", 42)

	ratios_total = train_ratio + validation_ratio + test_ratio
	if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or not 0 < test_ratio < 1:
		raise ValueError("train_ratio, val_ratio, and test_ratio must be between 0 and 1")
	if abs(ratios_total - 1.0) > 1e-6:
		raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1")

	normalization = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225],
	)
	train_transform = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalization,
	])
	evaluation_transform = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		normalization,
	])

	base_dataset = datasets.ImageFolder(dataset_path)
	labels = base_dataset.targets
	indices = list(range(len(base_dataset)))
	train_indices, remaining_indices = train_test_split(
		indices,
		test_size=validation_ratio + test_ratio,
		stratify=labels,
		random_state=seed,
	)

	remaining_labels = [labels[index] for index in remaining_indices]
	validation_fraction = validation_ratio / (validation_ratio + test_ratio)
	validation_indices, test_indices = train_test_split(
		remaining_indices,
		test_size=1 - validation_fraction,
		stratify=remaining_labels,
		random_state=seed,
	)

	train_dataset = Subset(
		datasets.ImageFolder(dataset_path, transform=train_transform),
		train_indices,
	)
	validation_dataset = Subset(
		datasets.ImageFolder(dataset_path, transform=evaluation_transform),
		validation_indices,
	)
	test_dataset = Subset(
		datasets.ImageFolder(dataset_path, transform=evaluation_transform),
		test_indices,
	)

	loader_kwargs = {
		"batch_size": batch_size,
		"num_workers": num_workers,
		"pin_memory": torch.cuda.is_available(),
	}
	train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
	validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
	test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

	return train_loader, validation_loader, test_loader, base_dataset.classes