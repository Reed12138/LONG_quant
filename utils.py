"""
辅助函数模块
包含数据转换、计算、日志等通用功能
"""

import pandas as pd
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # 设置日志级别
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转换为浮点数
    
    Args:
        value: 要转换的值
        default: 转换失败时的默认值
        
    Returns:
        float: 转换后的浮点数
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_change(old_value: float, new_value: float) -> float:
    """
    计算变化率
    
    Args:
        old_value: 旧值
        new_value: 新值
        
    Returns:
        float: 变化率（百分比）
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def format_price(price, symbol: str = "") -> str:
    try:
        # 强制转为 float
        price_float = float(price)
        return f"${price_float:.2f}"
    except (ValueError, TypeError):
        return str(price)  # 出错了就原样返回

def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    简化版数据验证
    
    Args:
        data: 数据DataFrame
        required_columns: 必需的列名列表
        
    Returns:
        bool: 数据是否有效
    """
    if data is None or data.empty:
        return False
    
    # 检查列是否存在
    for col in required_columns:
        if col not in data.columns:
            return False
    
    # 检查关键列是否有足够的数据
    if 'close' in data.columns:
        if len(data) < 20 or data['close'].isnull().any():
            return False
    
    return True

def calculate_returns(trades: List[Dict]) -> Dict:
    """
    计算交易回报统计
    
    Args:
        trades: 交易记录列表
        
    Returns:
        Dict: 统计信息
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0
        }
    
    closed_trades = [t for t in trades if t.get('action') == 'CLOSE']
    
    if not closed_trades:
        return {
            'total_trades': len(trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0
        }
    
    # 这里简化计算，实际需要根据开平仓计算盈亏
    winning_trades = 0
    losing_trades = 0
    
    return {
        'total_trades': len(closed_trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / len(closed_trades) * 100 if closed_trades else 0,
        'total_pnl': 0.0,  # 实际需要计算
        'avg_pnl': 0.0     # 实际需要计算
    }