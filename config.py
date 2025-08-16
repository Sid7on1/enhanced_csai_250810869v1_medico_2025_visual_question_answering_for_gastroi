import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'model': {
        'type': 'resnet50',
        'pretrained': True
    },
    'data': {
        'path': './data',
        'split': 'train'
    },
    'training': {
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001
    }
}

# Define an Enum for model types
class ModelType(Enum):
    RESNET50 = 'resnet50'
    VGG16 = 'vgg16'
    DENSENET121 = 'densenet121'

# Define a dataclass for configuration
@dataclass
class Config:
    model: Dict[str, str]
    data: Dict[str, str]
    training: Dict[str, float]

# Define a class for configuration management
class ConfigurationManager:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Config:
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return Config(**config)
        except FileNotFoundError:
            logger.warning(f'Config file not found: {self.config_file}')
            return Config(**DEFAULT_CONFIG)
        except yaml.YAMLError as e:
            logger.error(f'Error parsing config file: {e}')
            return Config(**DEFAULT_CONFIG)

    def save_config(self, config: Config):
        with open(self.config_file, 'w') as f:
            yaml.dump(config.__dict__, f, default_flow_style=False)

    def update_config(self, config: Config):
        self.config = config
        self.save_config(config)

# Define a class for configuration validation
class ConfigValidator:
    def __init__(self, config: Config):
        self.config = config

    def validate_model(self) -> bool:
        if self.config.model['type'] not in [m.value for m in ModelType]:
            logger.error('Invalid model type')
            return False
        return True

    def validate_data(self) -> bool:
        if not os.path.exists(self.config.data['path']):
            logger.error('Data path does not exist')
            return False
        return True

    def validate_training(self) -> bool:
        if self.config.training['batch_size'] <= 0:
            logger.error('Invalid batch size')
            return False
        if self.config.training['epochs'] <= 0:
            logger.error('Invalid number of epochs')
            return False
        if self.config.training['learning_rate'] <= 0:
            logger.error('Invalid learning rate')
            return False
        return True

    def validate(self) -> bool:
        return (self.validate_model() and
                self.validate_data() and
                self.validate_training())

# Define a class for configuration persistence
class ConfigPersistence:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file

    def save_config(self, config: Config):
        with open(self.config_file, 'w') as f:
            yaml.dump(config.__dict__, f, default_flow_style=False)

    def load_config(self) -> Config:
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return Config(**config)
        except FileNotFoundError:
            logger.warning(f'Config file not found: {self.config_file}')
            return Config(**DEFAULT_CONFIG)
        except yaml.YAMLError as e:
            logger.error(f'Error parsing config file: {e}')
            return Config(**DEFAULT_CONFIG)

# Define a class for configuration management
class ConfigManager:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config_validator = ConfigValidator(self.config_manager.config)
        self.config_persistence = ConfigPersistence()

    def get_config(self) -> Config:
        return self.config_manager.config

    def update_config(self, config: Config):
        self.config_manager.update_config(config)
        self.config_validator.validate()

    def save_config(self):
        self.config_persistence.save_config(self.config_manager.config)

# Create a ConfigManager instance
config_manager = ConfigManager()

# Load default config
config = config_manager.get_config()

# Print config
print(config)