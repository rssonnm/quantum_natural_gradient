import logging
import json
import os
from typing import Any, Dict, Optional

def setup_logging(level: int = logging.INFO) -> None:
    """
    Cài đặt hệ thống logging chuyên nghiệp cho toàn project.
    
    Args:
        level (int): Mức độ log (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_logger(name: str) -> logging.Logger:
    """
    Trả về một instance logger.
    """
    return logging.getLogger(name)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load cấu hình từ file JSON (hoặc YAML nếu cài thêm thư viện).
    
    Args:
        config_path (str): Đường dẫn file config.
        
    Returns:
        Dict[str, Any]: Dictionary chứa các tham số cấu hình.
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
