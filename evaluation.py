import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationException(Exception):
    """Base class for evaluation exceptions."""
    pass

class EvaluationConfig:
    """Configuration class for evaluation."""
    def __init__(self, batch_size: int = 32, num_workers: int = 4, device: str = 'cuda'):
        """
        Initialize evaluation configuration.

        Args:
        - batch_size (int): Batch size for data loading.
        - num_workers (int): Number of workers for data loading.
        - device (str): Device to use for evaluation (e.g., 'cuda' or 'cpu').
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

class EvaluationDataset(Dataset):
    """Dataset class for evaluation."""
    def __init__(self, data: List[Tuple[np.ndarray, int]], transform: callable = None):
        """
        Initialize evaluation dataset.

        Args:
        - data (List[Tuple[np.ndarray, int]]): List of tuples containing image data and labels.
        - transform (callable): Optional transform function to apply to image data.
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a data point from the dataset."""
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return torch.from_numpy(image), label

class Evaluator:
    """Evaluator class for model evaluation."""
    def __init__(self, model: torch.nn.Module, config: EvaluationConfig):
        """
        Initialize evaluator.

        Args:
        - model (torch.nn.Module): Model to evaluate.
        - config (EvaluationConfig): Evaluation configuration.
        """
        self.model = model
        self.config = config

    def evaluate(self, dataset: EvaluationDataset) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.

        Args:
        - dataset (EvaluationDataset): Dataset to evaluate on.

        Returns:
        - Dict[str, float]: Dictionary containing evaluation metrics.
        """
        try:
            # Create data loader
            data_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

            # Initialize metrics
            accuracy = 0
            precision = 0
            recall = 0
            f1 = 0
            total = 0

            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                for batch in data_loader:
                    images, labels = batch
                    images, labels = images.to(self.config.device), labels.to(self.config.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    accuracy += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
                    precision += precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                    recall += recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                    f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                    total += 1

            # Calculate average metrics
            accuracy /= total
            precision /= total
            recall /= total
            f1 /= total

            # Return metrics
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        except NotFittedError as e:
            logger.error(f"Model not fitted: {e}")
            raise EvaluationException("Model not fitted")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise EvaluationException("Error during evaluation")

class EvaluationMetrics:
    """Class for calculating evaluation metrics."""
    def __init__(self, true_labels: List[int], predicted_labels: List[int]):
        """
        Initialize evaluation metrics.

        Args:
        - true_labels (List[int]): True labels.
        - predicted_labels (List[int]): Predicted labels.
        """
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def accuracy(self) -> float:
        """Calculate accuracy."""
        return accuracy_score(self.true_labels, self.predicted_labels)

    def precision(self) -> float:
        """Calculate precision."""
        return precision_score(self.true_labels, self.predicted_labels, average='macro')

    def recall(self) -> float:
        """Calculate recall."""
        return recall_score(self.true_labels, self.predicted_labels, average='macro')

    def f1(self) -> float:
        """Calculate F1 score."""
        return f1_score(self.true_labels, self.predicted_labels, average='macro')

def main():
    # Create evaluation configuration
    config = EvaluationConfig(batch_size=32, num_workers=4, device='cuda')

    # Create dataset
    data = [(np.random.rand(224, 224, 3), 1) for _ in range(100)]
    dataset = EvaluationDataset(data)

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 5 * 5, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10)
    )

    # Create evaluator
    evaluator = Evaluator(model, config)

    # Evaluate model
    metrics = evaluator.evaluate(dataset)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()