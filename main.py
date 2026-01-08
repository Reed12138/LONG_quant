"""
加密货币量化交易主程序（实盘版）
5分钟级别交易系统，基于MACD + CCI + 数据库信号双重确认
使用 Gate.io API v4 实盘交易（USDT永续合约）
"""

import pandas as pd
import time
from typing import Dict
from datetime import datetime
import logging
import json
import requests
import hashlib
import hmac
import os

from collections import deque
from config import Config
from utils import setup_logger, format_price
from data_fetcher import DataFetcher
from risk_manager import RiskManager
from generate_signal import SignalGenerator  # 融合后的增强版

class CryptoTradingBot:
    """实盘加密货币交易机器人主类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("TradingBot", config.LOG_LEVEL)
        
        # 初始化模块
        self.data_fetcher = DataFetcher(config)
        self.risk_manager = RiskManager(config)
        self.signal_generator = SignalGenerator(config)
        
        # 数据存储
        self.market_data = {}
        self.trade_history = []
        self.max_unrealised_pnl = 0
        
        # 实盘API配置
        self.api_key = os.getenv('GATE_API_KEY')
        self.api_secret = os.getenv('GATE_API_SECRET')
        self.host = "https://api.gateio.ws"
        self.prefix = "/api/v4"
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        
        if not self.api_key or not self.api_secret:
            raise ValueError("请设置环境变量 GATE_API_KEY 和 GATE_API_SECRET")
        
        self.logger.info("实盘交易机器人初始化完成")
        self.logger.info(f"交易对: {config.SYMBOLS}")
        self.logger.info(f"K线周期: {config.INTERVAL}")

    # ================== Gate.io API 签名与请求 ==================
    @staticmethod
    def gen_sign(method: str, url: str, query_string: str = "", payload_string: str = ""):
        """
        Gate.io API v4 官方标准签名函数
        """
        key = os.getenv('GATE_API_KEY')
        secret = os.getenv('GATE_API_SECRET')

        if not key or not secret:
            raise ValueError("GATE_API_KEY 或 GATE_API_SECRET 未设置！")

        t = int(time.time())  # 必须是整数秒！

        m = hashlib.sha512()
        m.update(payload_string.encode('utf-8'))
        hashed_payload = m.hexdigest()

        sign_string = f"{method.upper()}\n{url}\n{query_string}\n{hashed_payload}\n{t}"

        sign = hmac.new(secret.encode('utf-8'), sign_string.encode('utf-8'), hashlib.sha512).hexdigest()

        return {
            'KEY': key,
            'Timestamp': str(t),
            'SIGN': sign
        }

    def _sign_request(self, method: str, url: str, query_string: str = None, payload_string: str = None):
        t = int(time.time())
        m = hashlib.sha512()
        m.update((payload_string or "").encode('utf-8'))
        hashed_payload = m.hexdigest()
        s = '%s\n%s\n%s\n%s\n%s' % (method, url, query_string or "", hashed_payload, t)
        sign = hmac.new(self.api_secret.encode('utf-8'), s.encode('utf-8'), hashlib.sha512).hexdigest()
        headers = self.headers.copy()
        headers.update({
            'KEY': self.api_key,
            'Timestamp': str(t),
            'SIGN': sign
        })
        return headers

    def _request(self, method: str, path: str, query_string: str = None, payload: dict = None):
        payload_str = json.dumps(payload, separators=(',', ':')) if payload else None
        headers = self._sign_request(method, self.prefix + path, query_string, payload_str)
        
        url = self.host + self.prefix + path
        if query_string:
            url += "?" + query_string
        
        try:
            response = requests.request(method, url, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                self.logger.error(f"API错误 {response.status_code}: {response.text}")
                return None
            return response.json()
        except Exception as e:
            self.logger.error(f"API请求异常: {e}")
            return None

    # ================== 账户与持仓查询（使用 gen_sign 方式） ==================

    def get_account_info(self) -> Dict:
        """
        获取 USDT 永续期货账户信息
        官方路径: GET /api/v4/futures/usdt/accounts
        """
        path = f"/futures/{self.config.SETTLE}/accounts"  # 通常 'usdt'

        # GET 请求，payload=None，query_string=""
        sign_headers = self.gen_sign('GET', self.prefix + path, "", "")

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        headers.update(sign_headers)

        url = self.host + self.prefix + path

        try:
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()

            if r.status_code != 200 or 'label' in data:
                label = data.get('label', 'UNKNOWN')
                msg = data.get('message', 'No message')
                self.logger.error(f"账户信息查询失败 [{label}]: {msg}")
                return {}

            # 提取关键字段并转为 float
            total = float(data.get('total', 0))
            available = float(data.get('available', 0))
            unrealised_pnl = float(data.get('unrealised_pnl', 0))
            position_margin = float(data.get('position_margin', 0))

            self.logger.debug(
                f"账户信息: 总权益 {total:.2f} USDT, "
                f"可用余额 {available:.2f} USDT, "
                f"未实现盈亏 {unrealised_pnl:+.2f} USDT, "
                f"仓位保证金 {position_margin:.2f} USDT"
            )

            return {
                'total': total,
                'available': available,
                'unrealised_pnl': unrealised_pnl,
                'position_margin': position_margin,
                'raw': data  # 保留原始数据用于调试
            }

        except Exception as e:
            self.logger.error(f"获取账户信息异常: {e}")
            return {}


    def get_position(self, contract: str) -> Dict | None:
        """
        获取指定合约的持仓信息（返回 dict 或 None）
        官方路径: GET /api/v4/futures/usdt/positions/{contract}
        """
        path = f"/futures/usdt/positions/{contract}"

        sign_headers = self.gen_sign('GET', self.prefix + path, "", "")

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        headers.update(sign_headers)

        url = self.host + self.prefix + path

        try:
            r = requests.get(url, headers=headers, timeout=10)
            result = r.json()

            if r.status_code != 200:
                label = result.get('label', 'UNKNOWN')
                msg = result.get('message', '')
                self.logger.warning(f"{contract} 持仓查询失败 [{label}]: {msg}")
                return None

            # Gate 返回 []（无仓位） 或 [ {...} ]（有仓位）
            if isinstance(result, list):
                if not result:  # 空列表 = 无持仓
                    self.logger.debug(f"{contract} 当前无持仓")
                    return None
                data = result[0]
            else:
                data = result  # 极少见情况

            # 统一转为 float，防止后续字符串比较错误
            try:
                data['size'] = float(data.get('size', '0'))
                data['entry_price'] = float(data.get('entry_price', '0'))
                data['unrealised_pnl'] = float(data.get('unrealised_pnl', '0'))
                data['value'] = float(data.get('value', '0'))
                data['margin'] = float(data.get('margin', '0'))
                data['liq_price'] = float(data.get('liq_price', '0'))
            except (ValueError, TypeError) as e:
                self.logger.error(f"{contract} 持仓数据转换失败: {e}")
                return None

            self.logger.debug(
                f"{contract} 持仓: {data['size']:.2f} 张 @ {data['entry_price']:.2f}, "
                f"未实现盈亏 {data['unrealised_pnl']:+.4f} USDT"
            )

            return data

        except Exception as e:
            self.logger.error(f"获取 {contract} 持仓异常: {e}")
            return None


    def get_contract_info(self, contract: str) -> Dict:
        """
        获取合约详细信息（主要用于获取 mark_price 等）
        官方路径: GET /api/v4/futures/usdt/contracts/{contract}
        """
        path = f"/futures/{self.config.SETTLE}/contracts/{contract}"

        sign_headers = self.gen_sign('GET', self.prefix + path, "", "")

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        headers.update(sign_headers)

        url = self.host + self.prefix + path

        try:
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()

            if r.status_code != 200 or 'label' in data:
                label = data.get('label', 'UNKNOWN')
                msg = data.get('message', '')
                self.logger.warning(f"{contract} 合约信息查询失败 [{label}]: {msg}")
                return {}

            mark_price = float(data.get('mark_price', 0))
            last_price = float(data.get('last_price', 0))

            self.logger.debug(f"{contract} 合约信息: mark_price={mark_price:.2f}, last_price={last_price:.2f}")

            return {
                'mark_price': mark_price,
                'last_price': last_price,
                'quanto_multiplier': float(data.get('quanto_multiplier', '0')),  # 合约面值，如 BTC 0.001
                'raw': data
            }

        except Exception as e:
            self.logger.error(f"获取 {contract} 合约信息异常: {e}")
            return {}

    # ================== 实盘交易执行 ==================
    def execute_live_trade(self, analysis_result: Dict):
        symbol = analysis_result['symbol']
        signal = analysis_result['signal']
        price = analysis_result['price']

        if signal not in ['BUY', 'SELL', 'CLEAR', 'HOLD']:
            return

        # 获取当前持仓
        pos_data = self.get_position(symbol)
        current_size = int(float(pos_data.get('size', 0))) if pos_data else 0

        # 风险清仓（优先级最高）
        if signal == 'CLEAR' and current_size != 0:
            self.logger.warning(f"🚨 触发风险清仓: {analysis_result['risk_reason']}")
            self.close_position(symbol, abs(current_size))
            return

        # 做多信号（双重确认后）
        if signal == 'BUY' and current_size == 0:
            account = self.get_account_info()
            available = float(account.get('available', 0))
            if available < 50:
                self.logger.warning("可用保证金不足50 USDT，跳过开仓")
                return

            size = int(self.config.SIZE)  # 固定张数开仓
            if size >= 1:
                self.logger.info(f"🚀 实盘开多: {symbol} {size}张")
                self.set_single_position_mode()                 # 设置单仓模式
                self.set_isolated_margin_mode(symbol)            # 设置逐仓模式
                self.set_leverage(symbol)                        # 设置杠杆
                result = self.open_position(symbol, size)
                if result is not None:                           # 开多
                    # self.logger.info(f"开仓成功: {symbol} {size}张")
                    pos_data = self.get_position(symbol)          # 获取最新持仓信息
                    entry_price = float(pos_data.get('entry_price', price))
                    stop_loss_price = entry_price * (1 - self.config.STOP_LOSS_PCT / 100)
                    self.set_stop_loss(symbol, stop_loss_price)   # 设置止损
                    self.logger.info(f"✅ 入场价格为: {entry_price}，止损价格已设置为 {stop_loss_price}，订单ID: {result.get('id')}")
            else:
                self.logger.info("计算张数不足1张，跳过开仓")

        elif signal == 'HOLD' and current_size > 0:
            self.logger.debug(f"持有信号（HOLD），当前持仓: {current_size}张")
            pos_data = self.get_position(symbol)
            unrealised_pnl = float(pos_data.get('unrealised_pnl', 0))
            position_value = float(pos_data.get('value', 0))  # 仓位名义价值（USDT）
            if position_value > 0.01:  # 避免除零
                unrealised_pnl_pct = (unrealised_pnl / position_value) * 100
            else:
                unrealised_pnl_pct = 0.0
            if unrealised_pnl_pct > self.config.HANDING_FEE_PCT:
                self.close_position(symbol, current_size)
                self.logger.info(f"💰 手续费获利平仓: {symbol} 浮盈 {unrealised_pnl_pct:.2f}% 超过手续费 {self.config.HANDING_FEE_PCT}%")

        # 卖出信号（平多）
        elif signal == 'SELL' and current_size > 0:
            self.logger.info(f"🔻 实盘平多: {symbol} {current_size}张")
            self.close_position(symbol, current_size)

    def set_single_position_mode(self, settle='usdt'):
        """
        设置为单仓模式（单向持仓模式）
        :param settle: 'usdt' 或 'btc'（默认 usdt 永续合约）
        :return: API 响应
        """
        host = "https://api.gateio.ws"
        prefix = "/api/v4"
        url = f'/futures/{settle}/dual_mode'
        query_param = 'dual_mode=false'  # false 表示单仓模式，true 表示双仓模式
        
        # 假设你的 gen_sign 函数签名类似：gen_sign(method, prefix + url, query_param, body='')
        # POST 请求 body 为空
        sign_headers = self.gen_sign('POST', prefix + url, query_param, '')
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        headers.update(sign_headers)
        
        full_url = host + prefix + url + "?" + query_param
        r = requests.post(full_url, headers=headers)  # 使用 POST 方法（body 为空）
        if r.status_code == 200:
                self.logger.info(f"✅ 持仓模式已设置为单向持仓")

        # print(r.json())
        return r.json()


    def set_stop_loss(self, contract: str, stop_price: float):
        """
        为当前多头仓位设置止损（价格触发订单）
        - 使用 mark_price 触发（price_type: 1，最公平、防操纵）
        - 触发后市价全平多头仓位（close-long-position + is_close: true）
        - 止损单永不过期

        参数:
            contract: 合约名，如 "BTC_USDT" 或 "ETH_USDT"
            stop_price: 止损触发价格（当 mark_price <= 此价格时触发）

        Returns:
            dict or None: 成功返回包含 'id' 的字典，失败返回 None
        """
        path = "/futures/usdt/price_orders"

        body = {
            "initial": {
                "contract": contract,
                "size": 0,                       # 0 = 不额外开仓
                "price": "0",                    # 市价平仓
                "tif": "ioc" ,                    # 市价单必须指定 ioc
                "close": True
            },
            "trigger": {
                "strategy_type": 0,              # 0 = 价格触发
                "price_type": 1,                 # 1 = mark_price（关键：使用标记价格）
                "price": f"{stop_price:.2f}",    # 触发价格，保留2位小数（足够）
                "rule": 2,                       # 2 = <= 触发（价格下跌时触发，多头止损）
                "expiration": 86400              # 过期时间：86400秒（1天），避免无限期挂单
            },
            "order_type": "close-long-position",  # 触发后自动全平当前多头仓位
            "is_close": True
        }

        payload_str = json.dumps(body, separators=(',', ':'))  # 紧凑格式用于签名

        # 使用你已有的官方标准 gen_sign 函数
        sign_headers = self.gen_sign('POST', self.prefix + path, "", payload_str)

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        headers.update(sign_headers)

        url = self.host + self.prefix + path

        try:
            r = requests.post(url, headers=headers, data=payload_str, timeout=10)

            if r.status_code == 200:
                result = r.json()
                if 'id' in result:
                    trigger_id = result['id']
                    self.logger.info(
                        f"✅ 止损设置成功: {contract} 当 mark_price <= {stop_price:.2f} 时自动全平多仓，"
                        f"触发订单ID: {trigger_id}"
                    )
                    return result
                else:
                    self.logger.error(f"止损返回异常（无ID）: {result}")
                    return None
            else:
                # 尝试解析错误信息
                try:
                    error = r.json()
                    label = error.get('label', 'UNKNOWN')
                    message = error.get('message', '')
                except:
                    label = 'HTTP_ERROR'
                    message = r.text or '空响应'
                self.logger.error(f"止损设置失败 [{label}]: {message}")
                return None

        except Exception as e:
            self.logger.error(f"止损请求异常: {e}")
            return None

    def set_isolated_margin_mode(self, contract: str):
        """
        设置指定合约为逐仓模式（Isolated）
        """
        path = f"/futures/usdt/positions/{contract}/margin_mode"
        body = {"mode": "isolated"}   # 小写即可，官方接受

        payload_str = json.dumps(body, separators=(',', ':'))  # 紧凑格式，无空格
        
        sign_headers = self.gen_sign('POST', self.prefix + path, "", payload_str)
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        headers.update(sign_headers)

        url = self.host + self.prefix + path
        try:
            r = requests.post(url, headers=headers, data=payload_str, timeout=10)
            result = r.json()
            if r.status_code == 200 and result.get('mode') == 'isolated':
                self.logger.info(f"✅ {contract} 已成功设置为逐仓模式")
                return True
            else:
                self.logger.warning(f"⚠️ 设置逐仓失败: {result}")
                return False
        except Exception as e:
            self.logger.error(f"设置逐仓异常: {e}")
            return False    
        
    def set_leverage(self, contract: str, leverage: int = None):
            """
            设置逐仓杠杆（关键：逐仓用 query 参数，不是 body！）
            """
            if leverage is None:
                leverage = self.config.LEVERAGE

            path = f"/futures/usdt/positions/{contract}/leverage"
            query_string = f"leverage={leverage}"  # ← 必须用 query！

            sign_headers = self.gen_sign('POST', self.prefix + path, query_string)
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            headers.update(sign_headers)

            url = self.host + self.prefix + path + "?" + query_string

            try:
                r = requests.post(url, headers=headers, timeout=10)
                result = r.json()
                if r.status_code == 200:
                    self.logger.info(f"杠杆设置成功: {contract} → {leverage}x")
                    return True
                else:
                    label = result.get('label', 'UNKNOWN')
                    msg = result.get('message', '')
                    self.logger.warning(f"杠杆设置失败 [{label}]: {msg}")
                    return False
            except Exception as e:
                self.logger.error(f"杠杆设置异常: {e}")
                return False
    
    def open_position(self, contract: str, size: int):
        """
        市价开多（修复 text 参数）
        """
        path = "/futures/usdt/orders"
        body = {
            "contract": contract,
            "size": str(size),
            "price": "0",
            "tif": "ioc",
            "text": f"t-long-{int(time.time())}"  # ← 必须以 t- 开头！
        }

        payload_str = json.dumps(body, separators=(',', ':'))

        sign_headers = self.gen_sign('POST', self.prefix + path, "", payload_str)
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        headers.update(sign_headers)

        url = self.host + self.prefix + path
        try:
            r = requests.post(url, headers=headers, data=payload_str, timeout=10)
            if r.status_code in [200, 201]:  # ← 关键：添加 201
                result = r.json()
                # 市价单成功条件
                if result.get('status') == 'finished' and int(result.get('left', 1)) == 0:
                    self.logger.info(f"✅ 开多成功: {contract} {size}张，成交价: {result.get('fill_price')}，订单ID: {result.get('id')}")
                    return result
                else:
                    self.logger.error(f"❌ 开仓未完全成交: {result}")
                    return None
            else:
                self.logger.error(f"❌ 开仓HTTP失败 {r.status_code}: {r.text}")
                return None

        except Exception as e:
            self.logger.error(f"开仓请求异常: {e}")
            return None

    def close_position(self, contract: str, size: int):
        """
        市价平仓（符合 Gate.io 最新 text 规则）
        """
        if size == 0:
            self.logger.info(f"{contract} 持仓为0，无需平仓")
            return True

        path = f"/futures/{self.config.SETTLE}/orders"

        body = {
            "contract": contract,
            "size": 0,                             # 单仓模式下平仓，size设0
            "close" : True,                        # 单仓模式下必须加 close=True
            "reduce_only": True,                   # 仅减仓
            "price": "0",                          # 市价单
            "tif": "ioc",                          # 立即成交或取消
            "text": f"t-bot_close_{contract}"      # 必须以 t- 开头！
        }

        payload_str = json.dumps(body, separators=(',', ':'))

        sign_headers = self.gen_sign('POST', self.prefix + path, "", payload_str)

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        headers.update(sign_headers)

        url = self.host + self.prefix + path

        try:
            r = requests.post(url, headers=headers, data=payload_str, timeout=10)
            result = r.json()

            if r.status_code == 201:
                order_id = result.get('id', 'N/A')
                status = result.get('status', 'unknown')
                self.logger.info(f"✅ 平仓成功: {contract}全部平仓{-size}张，订单ID: {order_id}，状态: {status}")
                return True
            else:
                label = result.get('label', 'UNKNOWN')
                message = result.get('message', 'No message')
                self.logger.error(f"❌ 平仓下单失败 [{label}]: {message}")
                return False

        except requests.exceptions.RequestException as e:
            self.logger.error(f"平仓网络异常: {e}")
            return False
        except Exception as e:
            self.logger.error(f"平仓异常: {e}")
            return False

    # ================== 其他原有方法（保持不变） ==================
    def test_connection(self) -> bool:
        price = self.data_fetcher.fetch_current_price("ETH_USDT")
        if price:
            self.logger.info(f"API连接成功！ETH当前价格: ${price:.2f}")
            return True
        return False

    def initialize_symbol(self, symbol: str) -> bool:
        """
        初始化交易对数据 - 简化版
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"正在初始化{symbol}...")
            
            # 获取历史数据
            df = self.data_fetcher.fetch_historical_data(symbol, self.config.LOOKBACK_PERIODS)
            
            # print(df.tail())
            if df is None or df.empty:
                self.logger.error(f"无法获取{symbol}历史数据")
                return False
            
            self.logger.info(f"获取到{len(df)}条{symbol}历史数据")
            
            # 计算技术指标
            df = self.data_fetcher.calculate_macd(df)
            df = self.data_fetcher.calculate_cci(df)
            df = self.data_fetcher.calculate_volume_ma(df)
            
            # 去掉前期的NaN行（因为指标计算需要历史数据）
            # 找到第一个所有指标都有值的行
            indicator_columns = ['macd', 'macd_signal', 'cci', 'volume_ma']
            
            # 确保所有指标列都存在
            for col in indicator_columns:
                if col not in df.columns:
                    self.logger.error(f"缺少指标列: {col}")
                    return False
            
            # 找到第一个所有指标都有值的行
            valid_mask = df[indicator_columns].notnull().all(axis=1)
            
            if not valid_mask.any():
                self.logger.error(f"没有完全有效的指标数据行")
                return False
            
            first_valid_idx = valid_mask.idxmax()  # 第一个True的索引
            df = df.loc[first_valid_idx:].copy()
            
            self.logger.info(f"去掉前期NaN数据，从 {first_valid_idx} 开始，保留 {len(df)} 条数据")
            
            if len(df) < 50:  # 确保有足够的数据进行交易
                self.logger.error(f"数据不足: {len(df)} 条")
                return False
            
            # 填充任何剩余的NaN值
            df = df.ffill().bfill()
            
            # 验证清理后的数据
            required_columns = ['open', 'high', 'low', 'close', 'volume', 
                               'macd', 'macd_signal', 'cci', 'volume_ma']
            
            # 简单验证：检查所有必需列是否存在
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.error(f"缺少列: {missing_cols}")
                return False
            
            # 检查是否有足够的有效数据
            for col in ['close', 'macd', 'macd_signal']:
                if df[col].isnull().any():
                    self.logger.error(f"列 {col} 仍有NaN值")
                    return False
            
            # 显示最新数据信息
            if len(df) > 0:
                latest = df.iloc[-1]
                self.logger.info(f"{symbol}最新数据 - 价格: ${latest['close']:.2f}, "
                               f"MACD: {latest['macd']:.4f}, "
                               f"信号线: {latest['macd_signal']:.4f}, "
                               f"CCI: {latest['cci']:.2f}")
            
            # 存储数据
            self.market_data[symbol] = df
            
            self.logger.info(f"{symbol}初始化成功，共{len(df)}条数据")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化{symbol}失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def update_market_data(self):
        """更新所有交易对的市场数据"""
        for symbol in self.config.SYMBOLS:
            try:
                if symbol not in self.market_data:
                    self.logger.warning(f"{symbol}数据未初始化，重新初始化...")
                    self.initialize_symbol(symbol)
                    continue
                
                # 更新数据
                existing_df = self.market_data[symbol]
                updated_df = self.data_fetcher.update_latest_data(symbol, existing_df)
                
                # 步骤2：先追加/更新实时价格，得到带当前K线的 df
                df_with_live = self._append_or_update_live_bar(
                    df=updated_df.copy(),
                    symbol=symbol,
                    timeframe=self.config.INTERVAL
                )
                updated_df = df_with_live

                if updated_df is not None and not updated_df.empty:
                    # 重新计算指标
                    updated_df = self.data_fetcher.calculate_macd(updated_df)
                    updated_df = self.data_fetcher.calculate_cci(updated_df)
                    updated_df = self.data_fetcher.calculate_volume_ma(updated_df)
                    updated_df = self.data_fetcher.calculate_adx_dmi(updated_df)
                    updated_df = self.data_fetcher.calculate_rsi(updated_df)
                    
                    self.market_data[symbol] = updated_df
                    
                    # 检查是否有新数据
                    if len(updated_df) > len(existing_df) or updated_df.index[-1] > existing_df.index[-1]:
                        self.logger.info(f"{symbol}数据更新成功，最新时间: {updated_df.index[-1]}")
                        # 显示最新指标
                        latest = updated_df.iloc[-1]
                        self.logger.debug(f"{symbol}最新指标 - 价格: ${latest['close']:.2f}, "
                                        f"MACD: {latest['macd']:.4f}, "
                                        f"信号线: {latest['macd_signal']:.4f}")
                else:
                    self.logger.warning(f"{symbol}数据更新失败")
                    
            except Exception as e:
                self.logger.error(f"更新{symbol}数据失败: {e}")
    
    def _append_or_update_live_bar(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """更新实时K线数据，添加当前未完成的K线"""
        df = df.copy()

        # 获取实时标记价格
        contract_info = self.get_contract_info(symbol)
        if contract_info and 'mark_price' in contract_info:
            current_price = float(contract_info['mark_price'])
        else:
            current_price = df['close'].iloc[-1]

        minutes = int(timeframe.rstrip('m'))
        now = pd.Timestamp.now(tz=df.index.tz)
        # 修改后（推荐，兼容未来版本）
        current_bar_start = now.floor(f'{minutes}min')
        last_bar_start = df.index[-1].floor(f'{minutes}min')

        if current_bar_start > last_bar_start:
            # 新开一根K线
            new_row = pd.Series({
                'open':  current_price,
                'high':  current_price,
                'low':   current_price,
                'close': current_price,
                'volume': 0,
            }, name=current_bar_start)
            df = pd.concat([df, new_row.to_frame().T])

        # 更新当前K线的 close/high/low
        last_idx = df.index[-1]
        df.loc[last_idx, 'close'] = current_price
        df.loc[last_idx, 'high'] = max(df.loc[last_idx, 'high'], current_price)
        df.loc[last_idx, 'low']  = min(df.loc[last_idx, 'low'],  current_price)

        # volume 可以考虑从 ticker 获取实时成交量，这里先简单设0或继承

        return df

    def analyze_symbol(self, symbol: str) -> Dict:
        """
        分析单个交易对，返回包含最近三个信号的历史
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signal': 'HOLD',                    # 当前（最新）信号
            'signal_history': ['HOLD', 'HOLD', 'HOLD'],  # 最近三个信号（最新在最后）
            'reason': '数据异常',
            'risk_clear': False,
            'risk_reason': '',
            'price': 0.0,
            'details': {}
        }
        
        try:
            if symbol not in self.market_data:
                self.logger.warning(f"{symbol}数据未初始化")
                self._update_signal_history(symbol, result['signal'])
                result['signal_history'] = list(self.signal_history.get(symbol, deque(['HOLD']*3, maxlen=3)))
                return result
            
            df = self.market_data[symbol]
            if len(df) < 20:
                self.logger.warning(f"{symbol}数据不足，只有{len(df)}条")
                self._update_signal_history(symbol, result['signal'])
                result['signal_history'] = list(self.signal_history.get(symbol, deque(['HOLD']*3, maxlen=3)))
                return result
            
            # 获取当前价格
            contract_info = self.get_contract_info(symbol)
            current_price = contract_info.get('mark_price') if contract_info else None
            if current_price is None:
                current_price = df['close'].iloc[-1]
                self.logger.debug(f"使用最新价格作为{symbol}风控价格: ${current_price:.2f}")
            else:
                # 改为
                df.loc[df.index[-1], 'close'] = float(current_price) # 使用gate标记价格作为风控价格
            
            result['price'] = current_price
            
            # 获取最新指标
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else latest
            
            # ============ 风险检查 ============
            # 1. 检查MACD背离
            divergence_detected, divergence_reason = self.risk_manager.detect_macd_divergence(df, symbol)
            
            # 2. 检查成交量异常
            # volume_anomaly, volume_reason = self.risk_manager.check_volume_anomaly(df, symbol)
            
            # 3. 检查持仓风险（如果有持仓）
            position = self.get_position(symbol)
            if position and float(position.get('size', 0)) != 0:
                unrealised_pnl = float(position.get('unrealised_pnl', 0))
                position_value = float(position.get('value', 0))  # 仓位名义价值（USDT）

                if position_value > 0.01:  # 避免除零
                    unrealised_pnl_pct = (unrealised_pnl / position_value) * 100
                else:
                    unrealised_pnl_pct = 0.0

                self.logger.debug(f"{symbol} 未实现盈亏: {unrealised_pnl:.4f} USDT ({unrealised_pnl_pct:+.2f}%)，仓位价值: {position_value:.2f} USDT")

                # 更新历史最大浮盈（用于移动止损）
                if unrealised_pnl_pct > self.max_unrealised_pnl :
                    self.max_unrealised_pnl = unrealised_pnl_pct
                    self.logger.debug(f"更新 {symbol} 最大浮盈记录: {unrealised_pnl_pct:.2f}%")

                # 风险条件检查
                risk_triggers = []

                # 固定止损 已经在开仓时设置止损价格 无需监控固定止损
                # if unrealised_pnl_pct <= -self.config.STOP_LOSS_PCT:
                #     risk_triggers.append((True, f"固定止损触发: 浮亏 {unrealised_pnl_pct:.2f}% ≤ -{self.config.STOP_LOSS_PCT}%"))

                # 固定止盈
                if unrealised_pnl_pct >= self.config.TAKE_PROFIT_PCT:
                    risk_triggers.append((True, f"固定止盈触发: 浮盈 {unrealised_pnl_pct:.2f}% ≥ {self.config.TAKE_PROFIT_PCT}%"))

                # 移动止损
                # max_pnl = self.max_unrealised_pnl
                # if max_pnl > 10 and unrealised_pnl < max_pnl * (1 - self.config.TRAILING_STOP_PCT/100): # 盈利阈值10美元 设置最大盈利阈值避免频繁触发
                #     risk_triggers.append((True, f"移动止损触发: 浮盈回撤 ${max_pnl - unrealised_pnl_pct:.2f}"))

                # MACD 背离等其他风险...
                if divergence_detected:
                    risk_triggers.append((True, f"MACD背离: {divergence_reason}"))

                # 如果任意风险触发 → CLEAR
                for triggered, reason in risk_triggers:
                    if triggered:
                        result['risk_clear'] = True
                        result['risk_reason'] = reason
                        result['signal'] = 'CLEAR'
                        result['reason'] = f"风险控制平仓: {reason}"
                        self.logger.warning(f"🚨 {symbol} 触发风险清仓: {reason}")

                        # 可选：在这里直接执行平仓，或留给 execute_live_trade 处理
                        # self.close_position(symbol, abs(float(position['size'])))

                        return result
            
            # ============ 如果没有风险，生成交易信号 ============
            # 如果触发风险清仓
            if result['risk_clear']:
                current_signal = 'CLEAR'
            else:
                # 生成交易信号
                signal, reason, details = self.signal_generator.generate_signal(symbol, df, current_price)
                current_signal = signal
                result['reason'] = reason
                result['details'] = details

            # ========== 关键修改：更新信号并记录历史 ==========
            result['signal'] = current_signal
            
            # 更新历史信号（使用 deque 限制长度）
            self._update_signal_history(symbol, current_signal)
            
            # 获取最近三个信号（旧 → 新）
            history_deque = self.signal_history.get(symbol, deque(['HOLD'] * 3, maxlen=3))
            result['signal_history'] = list(history_deque)  # 转为列表返回
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析{symbol}失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            result['reason'] = f"分析异常: {e}"
            return result

    def _update_signal_history(self, symbol: str, signal: str):
        """维护每个symbol的信号历史（最近N个）"""
        if not hasattr(self, 'signal_history'):
            self.signal_history = {}
        if symbol not in self.signal_history:
            self.signal_history[symbol] = deque(maxlen=3)  # 自动保留最近3个
        self.signal_history[symbol].append(signal)

    def update_account_value(self):
        """实盘版本：直接从API获取真实价值"""
        account = self.get_account_info()
        if account:
            print(f"实时账户: 总权益 ${float(account.get('total', 0)):.2f} | "
                  f"可用 ${float(account.get('available', 0)):.2f} | "
                  f"未实现盈亏 ${float(account.get('unrealised_pnl', 0)):+.2f}")

    def print_trading_summary(self):
        """打印真实账户摘要"""
        try:
            account = self.get_account_info()
            if not account:
                return

            print("\n" + "="*60)
            print(f"实盘账户摘要 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            print(f"总权益: ${float(account.get('total', 0)):.2f}")
            print(f"可用保证金: ${float(account.get('available', 0)):.2f}")
            print(f"未实现盈亏: ${float(account.get('unrealised_pnl', 0)):+.2f}")
            # print(f"已用保证金: ${float(account.get('position_initial_margin', 0)):.2f}")

            # 当前持仓
            for symbol in self.config.SYMBOLS:
                pos = self.get_position(symbol)
                if pos and float(pos.get('size', 0)) != 0:
                    size = float(pos['size'])
                    side = "多" if size > 0 else "空"
                    entry = float(pos['entry_price'])
                    unrealised = float(pos.get('unrealised_pnl', 0))
                    print(f"持仓: {symbol} {side} {abs(size)}张 @ ${entry:.2f} (浮盈 ${unrealised:+.2f})")

            print("="*60 + "\n")
        except Exception as e:
            self.logger.error(f"打印摘要失败: {e}")

    def trading_cycle(self):
        try:
            self.logger.info("="*60)
            self.logger.info(f"开始交易周期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            self.update_market_data()

            for symbol in self.config.SYMBOLS:
                analysis_result = self.analyze_symbol(symbol)

                # 日志输出
                signal_color = {"BUY": "🟢", "SELL": "🔴", "CLEAR": "🟡"}.get(analysis_result['signal'], "⚪")
                self.logger.info(
                    f"{signal_color} {symbol}: 价格={format_price(analysis_result['price'], symbol)}, "
                    f"信号={analysis_result['signal']}, 理由={analysis_result['reason']}"
                )
                with open("signals.log", "a", encoding="utf-8") as f:
                    f.write(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {symbol} | "
                        f"价格: {format_price(analysis_result['price'], symbol)} | "
                        f"信号: {analysis_result['signal']} | 理由: {analysis_result['reason']}\n"
                    )

                # 实盘执行
                self.execute_live_trade(analysis_result)

            self.update_account_value()
            if self.config.PRINT_TRADE_SUMMARY:
                self.print_trading_summary()

        except Exception as e:
            self.logger.error(f"交易周期执行失败: {e}", exc_info=True)

    def run(self):
        self.logger.info("启动实盘交易机器人...")
        if not self.test_connection():
            self.logger.error("API连接失败，退出")
            return

        # 初始化数据
        for symbol in self.config.SYMBOLS:
            if not self.initialize_symbol(symbol):
                self.logger.error(f"{symbol} 初始化失败")

        try:
            cycle_count = 0
            while True:
                cycle_count += 1
                self.logger.info(f"\n第 {cycle_count} 个交易周期")
                self.trading_cycle()

                print(f"\n{self.config.INTERVAL_SECONDS}秒后进入下一个周期...\n")
                time.sleep(self.config.INTERVAL_SECONDS)

        except KeyboardInterrupt:
            self.logger.info("\n手动停止实盘策略")
        except Exception as e:
            self.logger.error(f"主循环异常: {e}", exc_info=True)

def main():
    config = Config()
    bot = CryptoTradingBot(config)
    bot.run()

if __name__ == "__main__":
    main()