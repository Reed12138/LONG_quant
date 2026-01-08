"""
数据获取模块
负责从Gate.io API获取历史数据和实时数据
无需API密钥，直接调用公共接口
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from config import Config
from utils import safe_float, setup_logger

class DataFetcher:
    """Gate.io数据获取器"""
    
    def __init__(self, config: Config):
        """
        初始化数据获取器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = setup_logger("DataFetcher", config.LOG_LEVEL)
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def _format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号为Gate.io API要求的格式
        
        Args:
            symbol: 原始交易对符号，如 "BTC_USDT"
            
        Returns:
            str: 格式化后的交易对符号，如 "BTC_USDT"
        """
        # Gate.io API要求格式为 "BTC_USDT"（带下划线）
        # 如果已经是这个格式，直接返回
        if '_' in symbol:
            return symbol
        # 如果是 "BTCUSDT" 格式，添加下划线
        return f"{symbol[:3]}_{symbol[3:]}"
    
    def fetch_historical_data(self, symbol: str, limit: int = None) -> Optional[pd.DataFrame]:
        """
        获取 USDT 永续合约历史K线数据（无需认证，符合官方文档）
        
        Args:
            symbol: 合约名，如 "BTC_USDT" 或 "ETH_USDT"
            limit: 数据条数限制（最大2000，官方建议）
            
        Returns:
            pd.DataFrame: 列: open, high, low, close, volume，索引为 timestamp
        """
        if limit is None:
            limit = self.config.LOOKBACK_PERIODS
        limit = min(limit, 2000)  # 官方期货接口最大约2000条

        try:
            # 官方路径：USDT永续合约K线
            url = f"{self.config.API_BASE_URL}/futures/usdt/candlesticks"
            
            formatted_symbol = self._format_symbol(symbol)
            
            params = {
                'contract': formatted_symbol,
                'interval': self.config.INTERVAL,
                'limit': limit
            }
            
            self.logger.debug(f"请求永续合约历史K线 URL: {url} 参数: {params}")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                self.logger.warning(f"未获取到{symbol}永续合约历史数据，返回: {data}")
                return None
            
            self.logger.debug(f"获取到{len(data)}条{symbol}永续合约K线数据")

            # 官方返回格式（7列对象数组，非字符串数组）：
            # [{"t":秒时间戳, "o":开, "h":高, "l":低, "c":收, "v":合约张数}, ...]
            # 字段全为字符串
            df = pd.DataFrame(data)
            
            # 官方字段名正是 o, h, l, c, v, t
            df = df[['t', 'o', 'h', 'l', 'c', 'v']].copy()
            df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }, inplace=True)
            
            # 时间戳转 datetime 并设为索引
            df['timestamp'] = pd.to_datetime(df['t'].astype(int), unit='s')
            df.set_index('timestamp', inplace=True)
            df.drop(columns=['t'], inplace=True)
            
            # 转换数值类型（字符串 → float）
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)  # 官方已是字符串数字，直接astype即可
            
            # 官方返回最新在前，反转为时间升序
            df.sort_index(inplace=True)
            
            self.logger.info(
                f"获取{symbol}永续合约历史数据成功，共{len(df)}条记录，"
                f"时间范围: {df.index[0]} 到 {df.index[-1]}"
            )
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"网络错误获取{symbol}永续合约历史数据: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    self.logger.error(f"错误详情: {e.response.json()}")
                except:
                    self.logger.error(f"响应内容: {e.response.text[:200]}")
            return None
        except Exception as e:
            self.logger.error(f"获取{symbol}永续合约历史数据失败: {e}")
            return None
    
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """
        获取 USDT 永续合约当前价格（无需认证，符合官方文档）
        
        Args:
            symbol: 合约名，如 "BTC_USDT"
            
        Returns:
            float: 优先返回标记价格（mark_price），其次最新成交价
        """
        try:
            # 官方路径：USDT永续合约ticker
            url = f"{self.config.API_BASE_URL}/futures/usdt/tickers"
            
            formatted_symbol = self._format_symbol(symbol)
            
            params = {'contract': formatted_symbol}
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # 指定contract时返回单个dict，未指定返回list
            if isinstance(data, dict):
                tickers = [data]
            elif isinstance(data, list):
                tickers = data
            else:
                self.logger.warning(f"无法解析{symbol}永续合约ticker数据: {data}")
                return None
            
            for ticker in tickers:
                if ticker.get('contract') == formatted_symbol:
                    # 优先使用 mark_price（官方推荐用于风控、强平计算）
                    if ticker.get('mark_price'):
                        price = float(ticker['mark_price'])
                        self.logger.debug(f"{symbol} 永续合约标记价格: ${price}")
                        return price
                    
                    # fallback 到 last_price
                    if ticker.get('last_price'):
                        price = float(ticker['last_price'])
                        self.logger.debug(f"{symbol} 永续合约最新成交价: ${price}")
                        return price
                    
                    break
            
            self.logger.warning(f"未在响应中找到{symbol}的ticker数据: {data}")
            return None
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"网络错误获取{symbol}永续合约价格: {e}")
            return None
        except Exception as e:
            self.logger.error(f"获取{symbol}永续合约价格失败: {e}")
            return None
    
    def test_api_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 尝试获取一个常见的交易对信息
            test_symbol = "BTC_USDT"
            
            url = f"{self.config.API_BASE_URL}/spot/tickers"
            params = {'currency_pair': self._format_symbol(test_symbol)}
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info("API连接测试成功")
                return True
            else:
                self.logger.warning("API连接测试失败：无数据返回")
                return False
                
        except Exception as e:
            self.logger.error(f"API连接测试失败: {e}")
            return False
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            df: 包含价格数据的DataFrame
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            pd.DataFrame: 添加了MACD指标的数据
        """
        if fast is None:
            fast = self.config.MACD_FAST_LENGTH
        if slow is None:
            slow = self.config.MACD_SLOW_LENGTH
        if signal is None:
            signal = self.config.MACD_SIGNAL_LENGTH
        
        # 确保有足够的数据
        if len(df) < slow:
            self.logger.warning(f"数据不足计算MACD，需要至少{slow}条，当前{len(df)}条")
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_histogram'] = 0
            df['macd_slope'] = 0
            df['signal_slope'] = 0
            return df
        
        try:
            # 计算EMA
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            
            # 计算MACD线
            df['macd'] = exp1 - exp2
            
            # 计算信号线
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
            
            # 计算MACD柱状图
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 计算MACD斜率
            df['macd_slope'] = self.rolling_slope(df['macd'], window=self.config.MACD_SLOPE_LOOKBACK)      
            df['signal_slope'] = self.rolling_slope(df['macd_signal'], window=self.config.MACD_SLOPE_LOOKBACK)  
            
        except Exception as e:
            self.logger.error(f"计算MACD指标失败: {e}")
            # 设置默认值
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_histogram'] = 0
            df['macd_slope'] = 0
            df['signal_slope'] = 0
        
        return df
    
    @staticmethod
    def rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """
        计算滚动线性回归斜率
        window: 回看K线数量，30m周期建议5-8（即2.5-4小时）
        """
        window = Config.MACD_SLOPE_WINDOW
        def _slope(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]  # 返回斜率（一次多项式系数）
    
        return series.rolling(window).apply(_slope, raw=True)

    def calculate_cci(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        计算CCI指标
        
        Args:
            df: 包含价格数据的DataFrame
            period: CCI计算周期
            
        Returns:
            pd.DataFrame: 添加了CCI指标的数据
        """
        if period is None:
            period = self.config.CCI_PERIOD
        
        # 确保有足够的数据
        if len(df) < period:
            self.logger.warning(f"数据不足计算CCI，需要至少{period}条，当前{len(df)}条")
            df['cci'] = 0
            return df
        
        try:
            # 计算典型价格
            tp = (df['high'] + df['low'] + df['close']) / 3
            
            # 计算典型价格的简单移动平均
            sma_tp = tp.rolling(window=period).mean()
            
            # 计算平均偏差
            mean_deviation = abs(tp - sma_tp).rolling(window=period).mean()
            
            # 计算CCI（避免除零）
            cci_series = (tp - sma_tp) / (0.015 * mean_deviation.replace(0, np.nan))
            df['cci'] = cci_series.fillna(0)
            
        except Exception as e:
            self.logger.error(f"计算CCI指标失败: {e}")
            df['cci'] = 0
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        计算RSI指标（相对强弱指数）
        
        Args:
            df: 包含 'close' 列的DataFrame
            period: RSI计算周期
            
        Returns:
            pd.DataFrame: 添加了 'rsi' 列的数据
        """
        if period is None:
            period = self.config.RSI_PERIOD  # 建议默认14
        
        # 确保有足够的数据
        if len(df) < period + 1:  # 需要至少 period + 1 条数据才能计算差值
            self.logger.warning(f"数据不足计算RSI，需要至少{period + 1}条，当前{len(df)}条")
            df['rsi'] = 50.0  # 数据不足时返回中性值50，便于后续判断
            return df
        
        try:
            close = df['close']
            
            # 计算价格变化
            delta = close.diff()
            
            # 分离上涨和下跌
            gain = delta.where(delta > 0, 0.0)   # 上涨幅度
            loss = -delta.where(delta < 0, 0.0)  # 下跌幅度（取正值）
            
            # 第一段：使用简单移动平均作为初始值（Wilder原始方法）
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # 从第 period+1 行开始，使用Wilder平滑（当前值影响1/period，前值影响(period-1)/period）
            for i in range(period, len(df)):
                if i == period:  # 第一段已由rolling.mean计算
                    continue
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
            
            # 计算RS和RSI，避免除零
            rs = avg_gain / avg_loss.replace(0, np.nan)  # 当avg_loss为0时临时置nan
            rsi = 100 - (100 / (1 + rs))
            
            # 处理极端情况：当avg_loss为0（持续上涨）时，RSI=100；当avg_gain为0时，RSI=0
            rsi = rsi.fillna(100)  # 仅在avg_loss=0时填充100
            
            df['rsi'] = rsi.fillna(50.0)  # 前period行无法计算，填充中性值50
            
        except Exception as e:
            self.logger.error(f"计算RSI指标失败: {e}")
            df['rsi'] = 50.0
        
        return df

    def calculate_adx_dmi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        计算 +DI, -DI 和 ADX 指标（纯pandas实现，不依赖talib）
        
        Args:
            df: 包含 'high', 'low', 'close' 的DataFrame
            period: 计算周期，默认使用config中的值
            
        Returns:
            pd.DataFrame: 添加了 'plus_di', 'minus_di', 'adx' 列的数据
        """
        if period is None:
            period = self.config.ADX_PERIOD  
        
        # 确保有足够的数据（至少需要 period + 一些额外用于平滑）
        if len(df) < period + 10:
            self.logger.warning(f"数据不足计算ADX/DMI，需要至少{period + 10}条，当前{len(df)}条")
            df['plus_di'] = 0.0
            df['minus_di'] = 0.0
            df['adx'] = 0.0
            return df
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # 计算 True Range (TR)
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum.reduce([tr1, tr2, tr3])
            tr[0] = np.nan  # 第一行无前收盘
            
            # 计算 +DM 和 -DM
            plus_dm = np.diff(high)
            minus_dm = np.diff(low) * -1  # 取正值
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # 标准规则：当 +DM 和 -DM 同时大于0时，只保留较大者
            both_positive = (plus_dm > 0) & (minus_dm > 0)
            larger_up = plus_dm > minus_dm
            plus_dm[both_positive & ~larger_up] = 0
            minus_dm[both_positive & larger_up] = 0
            
            # 在开头补0以对齐长度
            plus_dm = np.concatenate([[0], plus_dm])
            minus_dm = np.concatenate([[0], minus_dm])
            tr = np.nan_to_num(tr, nan=0.0)  # 第一行设0避免后续问题
            
            # Wilder 平滑函数
            def wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
                """
                标准 Wilder 平滑（匹配 TradingView / Gate.io / TA-Lib）
                - 初始值：前 period 个值的 sum（放在 index period-1）
                - 后续：prev * (period-1) + current / period
                - 前 period-1 行设为 np.nan（图表通常不显示）
                """
                if len(series) < period:
                    return np.full_like(series, np.nan, dtype=float)
                
                smoothed = np.full_like(series, np.nan, dtype=float)
                
                # 初始值：sum of first period values（忽略 NaN 或已补0）
                first_sum = np.nansum(series[:period])  # 或 series[:period].sum() 如果已补0
                smoothed[period - 1] = first_sum  # 第一个有效值放在 index period-1
                
                # 后续 Wilder 平滑
                for i in range(period, len(series)):
                    prev = smoothed[i - 1]
                    if np.isnan(prev):
                        prev = smoothed[i - period]  # 备用（通常不会触发）
                    smoothed[i] = (prev * (period - 1) + series[i]) / period
                
                return smoothed
            
            atr = wilder_smooth(tr, period)
            plus_di_raw = wilder_smooth(plus_dm, period)
            minus_di_raw = wilder_smooth(minus_dm, period)
            
            # +DI 和 -DI
            plus_di = 100 * plus_di_raw / atr
            minus_di = 100 * minus_di_raw / atr
            
            # DX
            di_sum = plus_di_raw + minus_di_raw
            di_diff = np.abs(plus_di_raw - minus_di_raw)
            dx = np.where(di_sum == 0, 0, 100 * di_diff / di_sum)
            
            # ADX：对 DX 再次Wilder平滑
            adx = wilder_smooth(dx, period)
            
            # 赋值到DataFrame
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            df['adx'] = adx
            
        except Exception as e:
            self.logger.error(f"计算ADX/DMI指标失败: {e}")
            df['plus_di'] = 0.0
            df['minus_di'] = 0.0
            df['adx'] = 0.0
        
        return df

    def calculate_adx_dmi_safe(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        安全版本的ADX/DMI计算，完全避免NaN和警告
        """
        if period is None:
            period = self.config.ADX_PERIOD
        
        if len(df) < period:
            df['plus_di'] = 0.0
            df['minus_di'] = 0.0
            df['adx'] = 0.0
            return df
        
        # 使用原始计算方法
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Wilder平滑（修改版，避免NaN）
        def wilder_smooth_safe(series: np.ndarray, period: int) -> np.ndarray:
            n = len(series)
            smoothed = np.zeros(n, dtype=float)
            
            if n == 0:
                return smoothed
            
            # 使用简单平均初始化
            for i in range(min(period, n)):
                smoothed[i] = np.mean(series[:i+1]) if i > 0 else series[0]
            
            # Wilder递推
            for i in range(period, n):
                smoothed[i] = (smoothed[i-1] * (period - 1) + series[i]) / period
            
            return smoothed
        
        # 计算平滑值
        atr = wilder_smooth_safe(tr, period)
        plus_dm_s = wilder_smooth_safe(plus_dm, period)
        minus_dm_s = wilder_smooth_safe(minus_dm, period)
        
        # 计算DI（安全除法）
        epsilon = 1e-10
        atr_safe = np.maximum(atr, epsilon)
        
        plus_di = 100 * plus_dm_s / atr_safe
        minus_di = 100 * minus_dm_s / atr_safe
        
        plus_di = np.clip(plus_di, 0, 100)
        minus_di = np.clip(minus_di, 0, 100)
        
        # 计算DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        
        # 使用安全的逐元素计算
        dx = np.zeros_like(di_sum)
        for i in range(len(dx)):
            if di_sum[i] > epsilon:
                dx[i] = 100 * di_diff[i] / di_sum[i]
            else:
                dx[i] = 0.0
        
        # 计算ADX
        adx = wilder_smooth_safe(dx, period)
        adx = np.clip(adx, 0, 100)
        
        # 赋值回DataFrame
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx'] = adx
        
        return df

    def calculate_adx_dmi_gpt(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        计算 +DI, -DI 和 ADX 指标（纯 pandas / numpy 实现）
        对齐 Gate / TradingView / TA-Lib 的 Wilder 定义
        """
        if period is None:
            period = self.config.ADX_PERIOD

        if len(df) < period + 10:
            self.logger.warning(
                f"数据不足计算 ADX/DMI，需要至少 {period + 10} 条，当前 {len(df)} 条"
            )
            df['plus_di'] = 0.0
            df['minus_di'] = 0.0
            df['adx'] = 0.0
            return df

        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # ========= True Range =========
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum.reduce([tr1, tr2, tr3])
            tr[0] = 0.0  # 第一根 TR 明确设为 0（Gate / TV 行为）

            # ========= Directional Movement =========
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            # 对齐长度
            plus_dm = np.concatenate([[0.0], plus_dm])
            minus_dm = np.concatenate([[0.0], minus_dm])

            # ========= Wilder 平滑 =========
            def wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
                smoothed = np.full_like(series, np.nan, dtype=float)
                if len(series) < period:
                    return smoothed

                # 第一个有效值：前 period 项的和（index = period-1）
                smoothed[period - 1] = np.sum(series[:period])

                # Wilder 递推
                for i in range(period, len(series)):
                    smoothed[i] = (
                        smoothed[i - 1] * (period - 1) + series[i]
                    ) / period

                return smoothed

            atr = wilder_smooth(tr, period)
            plus_dm_s = wilder_smooth(plus_dm, period)
            minus_dm_s = wilder_smooth(minus_dm, period)

            # ========= +DI / -DI =========
            plus_di = 100 * plus_dm_s / atr
            minus_di = 100 * minus_dm_s / atr

            # ========= DX（⚠️ 必须基于 DI，而不是 DM） =========
            di_sum = plus_di + minus_di
            di_diff = np.abs(plus_di - minus_di)
            dx = np.where(di_sum == 0, 0.0, 100 * di_diff / di_sum)

            # ========= ADX =========
            adx = wilder_smooth(dx, period)

            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            df['adx'] = adx

        except Exception as e:
            self.logger.error(f"计算 ADX/DMI 指标失败: {e}")
            df['plus_di'] = 0.0
            df['minus_di'] = 0.0
            df['adx'] = 0.0

        return df

    def calculate_volume_ma(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        计算成交量移动平均
        
        Args:
            df: 数据DataFrame
            period: 移动平均周期
            
        Returns:
            pd.DataFrame: 添加了成交量MA的数据
        """
        if period is None:
            period = self.config.VOLUME_MA_PERIOD
        
        try:
            if len(df) >= period:
                df['volume_ma'] = df['volume'].rolling(window=period).mean()
            else:
                df['volume_ma'] = df['volume']
        except Exception as e:
            self.logger.error(f"计算成交量MA失败: {e}")
            df['volume_ma'] = df['volume']
        
        return df
    
    def update_latest_data(self, symbol: str, existing_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        更新最新数据
        
        Args:
            symbol: 交易对
            existing_df: 现有数据
            
        Returns:
            pd.DataFrame: 更新后的数据
        """
        try:
            # 获取最新数据（多取几条以防缺失）
            latest_df = self.fetch_historical_data(symbol, limit=30)
            
            if latest_df is None or latest_df.empty:
                self.logger.warning(f"无法获取{symbol}最新数据")
                return existing_df
            
            # 如果现有数据为空，直接返回最新数据
            if existing_df is None or existing_df.empty:
                return latest_df
            
            # 获取最新数据的时间戳
            latest_timestamp = latest_df.index[-1]
            
            # 如果最新数据的时间戳已经存在于现有数据中，直接返回现有数据
            if latest_timestamp in existing_df.index:
                self.logger.debug(f"{symbol}数据已是最新，无需更新")
                return existing_df
            
            # 合并数据
            combined_df = pd.concat([existing_df, latest_df])
            
            # 去重并保留最新的
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            
            # 只保留最近的数据
            if len(combined_df) > self.config.LOOKBACK_PERIODS:
                combined_df = combined_df.iloc[-self.config.LOOKBACK_PERIODS:]
            
            self.logger.debug(f"{symbol}数据更新完成，最新时间: {combined_df.index[-1]}")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"更新{symbol}数据失败: {e}")
            return existing_df