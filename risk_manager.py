"""
风险管理模块
负责风险控制：背离检测、成交量异常、止盈止损
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from config import Config
from utils import setup_logger

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Config):
        """
        初始化风险管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = setup_logger("RiskManager", config.LOG_LEVEL)
        
        # 存储持仓信息
        self.positions = {}
        
    def detect_macd_divergence(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
        """
        检测MACD顶背离
        
        Args:
            df: 包含价格和MACD的数据
            symbol: 交易对
            
        Returns:
            Tuple[bool, str]: (是否检测到背离, 描述信息)
        """
        try:
            if len(df) < self.config.DIVERGENCE_LOOKBACK:
                return False, "数据不足进行背离检测"
            
            # 获取最近的数据
            recent_df = df.tail(self.config.DIVERGENCE_LOOKBACK).copy()
            
            # 寻找价格高点
            price_high_idx = recent_df['high'].nlargest(2).index.tolist()
            macd_high_idx = recent_df['macd'].nlargest(2).index.tolist()
            
            if len(price_high_idx) < 2 or len(macd_high_idx) < 2:
                return False, "无法找到足够的峰值"
            
            # 检查顶背离（价格创新高，MACD未创新高）
            if (recent_df.loc[price_high_idx[0], 'high'] > recent_df.loc[price_high_idx[1], 'high'] and
                recent_df.loc[macd_high_idx[0], 'macd'] < recent_df.loc[macd_high_idx[1], 'macd']):
                
                price_diff = recent_df.loc[price_high_idx[0], 'high'] - recent_df.loc[price_high_idx[1], 'high']
                macd_diff = recent_df.loc[macd_high_idx[0], 'macd'] - recent_df.loc[macd_high_idx[1], 'macd']
                
                return True, f"顶背离检测：价格↑{price_diff:.2f}，MACD↓{macd_diff:.4f}"
            
            # 检查底背离（价格创新低，MACD未创新低）
            price_low_idx = recent_df['low'].nsmallest(2).index.tolist()
            macd_low_idx = recent_df['macd'].nsmallest(2).index.tolist()
            
            if len(price_low_idx) < 2 or len(macd_low_idx) < 2:
                return False, "无法找到足够的谷值"
            
            if (recent_df.loc[price_low_idx[0], 'low'] < recent_df.loc[price_low_idx[1], 'low'] and
                recent_df.loc[macd_low_idx[0], 'macd'] > recent_df.loc[macd_low_idx[1], 'macd']):
                
                price_diff = recent_df.loc[price_low_idx[0], 'low'] - recent_df.loc[price_low_idx[1], 'low']
                macd_diff = recent_df.loc[macd_low_idx[0], 'macd'] - recent_df.loc[macd_low_idx[1], 'macd']
                
                return True, f"底背离检测：价格↓{price_diff:.2f}，MACD↑{macd_diff:.4f}"
            
            return False, "未检测到明显背离"
            
        except Exception as e:
            self.logger.error(f"背离检测失败: {e}")
            return False, f"背离检测异常: {e}"
    
    def check_volume_anomaly(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
        """
        检查成交量异常
        
        Args:
            df: 包含成交量数据
            symbol: 交易对
            
        Returns:
            Tuple[bool, str]: (是否异常, 描述信息)
        """
        try:
            if len(df) < 5:
                return False, "数据不足进行成交量分析"
            
            # 计算最新成交量
            latest_volume = df['volume'].iloc[-1]
            
            # 计算成交量均线
            if 'volume_ma' not in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=self.config.VOLUME_MA_PERIOD).mean()
            
            current_ma = df['volume_ma'].iloc[-1]
            
            if pd.isna(current_ma) or current_ma == 0:
                return False, "成交量MA无效"
            
            # 计算成交量比率
            volume_ratio = latest_volume / current_ma
            
            # 检查成交量异常 暂不检测成交量
            # if volume_ratio > self.config.VOLUME_SPIKE_THRESHOLD:
            #     return True, f"成交量异常放大：{volume_ratio:.2f}倍于均量"
            # elif volume_ratio < self.config.VOLUME_DROP_THRESHOLD:
            #     return True, f"成交量异常萎缩：{volume_ratio:.2f}倍于均量"
            # else:
            #     return False, f"成交量正常：{volume_ratio:.2f}倍于均量"
                
        except Exception as e:
            self.logger.error(f"成交量检查失败: {e}")
            return False, f"成交量检查异常: {e}"
    
    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float) -> Tuple[bool, str]:
        """
        检查止损条件
        
        Args:
            symbol: 交易对
            entry_price: 入场价格
            current_price: 当前价格
            
        Returns:
            Tuple[bool, str]: (是否触发止损, 描述信息)
        """
        if entry_price <= 0:
            return False, "入场价格无效"
        
        # 计算价格变化百分比
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        # 多头止损（价格下跌超过止损比例）
        if price_change_pct < -self.config.STOP_LOSS_PCT:
            return True, f"触发止损：下跌{abs(price_change_pct):.2f}%"
        
        # 空头止损（价格上涨超过止损比例）
        elif price_change_pct > self.config.STOP_LOSS_PCT:
            return True, f"触发止损：上涨{abs(price_change_pct):.2f}%"
        
        return False, f"未触发止损：当前变化{price_change_pct:.2f}%"
    
    def check_take_profit(self, symbol: str, entry_price: float, current_price: float) -> Tuple[bool, str]:
        """
        检查止盈条件
        
        Args:
            symbol: 交易对
            entry_price: 入场价格
            current_price: 当前价格
            
        Returns:
            Tuple[bool, str]: (是否触发止盈, 描述信息)
        """
        if entry_price <= 0:
            return False, "入场价格无效"
        
        # 计算价格变化百分比
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        # 多头止盈（价格上涨超过止盈比例）
        if price_change_pct > self.config.TAKE_PROFIT_PCT:
            return True, f"触发止盈：上涨{price_change_pct:.2f}%"
        
        # 空头止盈（价格下跌超过止盈比例）
        elif price_change_pct < -self.config.TAKE_PROFIT_PCT:
            return True, f"触发止盈：下跌{abs(price_change_pct):.2f}%"
        
        return False, f"未触发止盈：当前变化{price_change_pct:.2f}%"
    
    def update_position(self, symbol: str, side: str, entry_price: float, quantity: float):
        """
        更新持仓信息
        
        Args:
            symbol: 交易对
            side: 方向（LONG/SHORT）
            entry_price: 入场价格
            quantity: 数量
        """
        self.positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': pd.Timestamp.now(),
            'highest_price': entry_price if side == 'LONG' else 0,
            'lowest_price': entry_price if side == 'SHORT' else float('inf')
        }
    
    def update_price_extremes(self, symbol: str, current_price: float):
        """
        更新价格极值
        
        Args:
            symbol: 交易对
            current_price: 当前价格
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            
            if pos['side'] == 'LONG':
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
            elif pos['side'] == 'SHORT':
                if current_price < pos['lowest_price']:
                    pos['lowest_price'] = current_price
    
    def clear_position(self, symbol: str):
        """
        清空持仓
        
        Args:
            symbol: 交易对
        """
        if symbol in self.positions:
            del self.positions[symbol]
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        获取持仓信息
        
        Args:
            symbol: 交易对
            
        Returns:
            Optional[Dict]: 持仓信息字典
        """
        return self.positions.get(symbol)