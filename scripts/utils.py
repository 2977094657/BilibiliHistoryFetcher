import yaml
import os

def get_base_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    base_path = get_base_path()
    config_path = os.path.join(base_path, 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
