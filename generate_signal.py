"""
增强版信号生成模块
融合数据库信号 + 技术指标双重确认
只有两者都看多时才开多，否则倾向平仓或持有
"""

import pandas as pd
import pymysql
import logging
import numpy as np
import os
from typing import Tuple, Dict
from datetime import datetime

from config import Config
from utils import setup_logger


class SignalGenerator:
    """增强版交易信号生成器（数据库 + 技术指标 双重确认）"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("SignalGenerator", config.LOG_LEVEL)
        
        # 数据库配置（建议从 config 读取，这里硬编码便于你直接运行）
        self.db_config = {
            'host': '47.108.187.39',
            'user': 'root',
            'password': os.getenv('DB_PASSWORD'),  # 推荐从 Config 读取
            'database': 'crypto_trading',
            'port': 3306,
            'charset': 'utf8mb4'
        }

        self.table_name = ["eth_sign", "btc_sign"]  # 同时监控 ETH 和 BTC

        # 信号状态（用于技术指标连续确认）
        self.signal_states = {}

    def _read_latest_db_signals(self, limit: int = 10) -> Tuple[str, str]:
        """
        读取数据库中 ETH 和 BTC 的最新信号（使用 SQLAlchemy，避免警告）
        
        新规则：
        - 如果任意币种最新3条信号均为“空头” → 返回 "SELL"（强制离场信号）
        - 如果 ETH 和 BTC 同时满足“空翻多”形态 → 返回 "BUY"（买入信号）
        - 否则 → 返回 "HOLD"
        
        “空翻多”形态定义（你原代码中的判断逻辑已修正为正确版本）：
            最近6条中：较新3条连续多头，较旧3条连续空头，且最新一条为多头
        
        Returns:
            (signal: str, reason: str)  # "BUY" / "SELL" / "HOLD"
        """
        from sqlalchemy import create_engine
        
        try:
            # 缓存 engine，避免重复创建（推荐做法）
            if not hasattr(self, 'engine'):
                connection_string = (
                    f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
                    f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
                    f"?charset={self.db_config['charset']}"
                )
                self.engine = create_engine(connection_string)
                self.logger.info("SQLAlchemy engine 创建成功")

            tables = ["eth_sign", "btc_sign"]  # 明确处理两个表
            results = {}

            for table in tables:
                coin = "BTC" if "btc" in table else "ETH"
                query = f"""
                SELECT zh_signal, created_at 
                FROM `{table}`
                ORDER BY created_at DESC
                LIMIT {limit}
                """
                df = pd.read_sql(query, self.engine)

                if len(df) < 7:
                    self.logger.warning(f"{coin} 信号不足（仅{len(df)}条），无法判断形态")
                    results[coin] = {
                        'is_reversal': False,
                        'latest_3_all_empty': False,
                        'directions': df['zh_signal'].tolist() if not df.empty else [],
                        'latest': df['zh_signal'].iloc[0] if not df.empty else None
                    }
                    continue

                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.sort_values('created_at', ascending=False).reset_index(drop=True)
                directions = df['zh_signal'].tolist()

                # 最新3条是否全为空头
                latest_3_all_empty = directions[:3] == ['空头', '空头', '空头']

                # 空翻多形态判断
                recent_6 = directions[:6]
                latest = directions[0]
                is_reversal = (
                    recent_6[1:3] == ['多头', '多头'] and   # 较新的2条：连续多头
                    recent_6[3:5] == ['空头', '空头'] and   # 较旧的2条：连续空头
                    latest == '多头'
                )

                results[coin] = {
                    'is_reversal': is_reversal,
                    'latest_3_all_empty': latest_3_all_empty,
                    'directions': directions[:7],
                    'latest': latest
                }

                self.logger.debug(
                    f"{coin} 信号序列（最新→旧）: {directions[:7]} | "
                    f"空翻多: {is_reversal} | 最新3条连续空头: {latest_3_all_empty}"
                )

            # ==================== 信号决策 ====================
            # 读取完成后，调用强多判断
            eth_dirs = results["ETH"]['directions']
            btc_dirs = results["BTC"]['directions']

            # 第一优先：连续3空离场（保持不变）
            sell_reasons = []
            for coin, dirs in [("ETH", eth_dirs), ("BTC", btc_dirs)]:
                if len(dirs) >= 3 and dirs[:3] == ['空头', '空头', '空头']:
                    sell_reasons.append(f"{coin}最新3条连续空头")

            if sell_reasons:
                reason = "；".join(sell_reasons)
                self.logger.warning(f"🚨 触发SELL信号: {reason}")
                return "SELL", reason

            # 第二优先：强多信号判断（新逻辑）
            is_strong_buy, buy_reason = self.is_strong_bullish_signal(eth_dirs, btc_dirs)
            if is_strong_buy:
                self.logger.info(f"🚀 数据库信号提示可BUY: {buy_reason}")
                return "BUY", buy_reason

            # 其他情况
            return "HOLD", "双币未达强多标准"

        except Exception as e:
            self.logger.error(f"读取数据库信号失败: {e}")
            return "HOLD", f"数据库异常: {str(e)}"
    
    def is_strong_bullish_signal(self, eth_directions: list, btc_directions: list) -> Tuple[bool, str]:
        """
        双币强多信号判断（多级优先级）
        
        优先级排序：
        1. 最高：双币同时经典空翻多
        2. 次高：ETH 4连多 + BTC 至少2连多（强势延续）
        3. 中等：双币同时3连多
        4. 低：任一币种经典空翻多（单币确认）
        
        Args:
            eth_directions: ETH 最新到最旧的信号列表（至少7条）
            btc_directions: BTC 最新到最旧的信号列表（至少7条）
        
        Returns:
            (is_strong_buy: bool, reason: str)  # "BUY" / "HOLD"
        """
        if len(eth_directions) < 7 or len(btc_directions) < 7:
            return False, "任一币种信号不足，无法判断强多形态"

        reason_parts = []
        signal_strength = 0  # 信号强度等级：1=低，2=中等，3=次高，4=最高

        # ================== 优先级1：双币经典空翻多（最高） ==================
        def is_classic_reversal(directions: list) -> bool:
            if len(directions) < 7:
                return False
            recent_6 = directions[:6]
            return (recent_6[0:3] == ['多头', '多头', '多头'] and  # 最新3条连续多头
                    recent_6[3:6] == ['空头', '空头', '空头'])       # 较旧3条连续空头

        eth_classic = is_classic_reversal(eth_directions)
        btc_classic = is_classic_reversal(btc_directions)

        if eth_classic and btc_classic:
            reason_parts.append("🔥 双币经典空翻多（最高优先级）")
            signal_strength = 4
            return True, "；".join(reason_parts)

        # ================== 优先级2：ETH 4连多 + BTC 至少2连多（次高） ==================
        eth_4_long = eth_directions[:4] == ['多头', '多头', '多头', '多头']
        btc_2_long = eth_directions[:2] == ['多头', '多头']
        btc_3_long = eth_directions[:3] == ['多头', '多头', '多头']  # 额外检查3连多

        if eth_4_long and (btc_2_long or btc_3_long):
            reason_parts.append("⚡ ETH 4连多 + BTC 2/3连多（强势延续）")
            signal_strength = 3
            return True, "；".join(reason_parts)

        # ================== 优先级3：双币3连多（中等） ==================
        eth_3_long = eth_directions[:3] == ['多头', '多头', '多头']
        btc_3_long = btc_directions[:3] == ['多头', '多头', '多头']

        if eth_3_long and btc_3_long:
            reason_parts.append("📈 双币3连多（中等多头）")
            signal_strength = 2
            return True, "；".join(reason_parts)

        # ================== 优先级4：任一币种经典空翻多（低优先级） ==================
        # if eth_classic:
        #     reason_parts.append("ETH 经典空翻多（单币确认）")
        #     signal_strength = 1
        # if btc_classic:
        #     reason_parts.append("BTC 经典空翻多（单币确认）")
        #     signal_strength = 1

        if signal_strength > 0:
            reason = "；".join(reason_parts)
            return True, reason

        # ================== 未满足任何条件 ==================
        hold_reasons = []
        if not eth_4_long:
            hold_reasons.append(f"ETH 仅{eth_directions[:4]}（非4连多）")
        if not btc_3_long and not btc_2_long:
            hold_reasons.append(f"BTC 仅{btc_directions[:3]}（非2/3连多）")
        if not eth_classic:
            hold_reasons.append("ETH 未空翻多")
        if not btc_classic:
            hold_reasons.append("BTC 未空翻多")

        return False, f"未达强多标准: {' | '.join(hold_reasons)}"

    def _technical_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Tuple[str, str, Dict]:
        """
        原有技术指标信号逻辑（保持不变，仅轻微简化异常处理）

        首先判断趋势,再生成信号
        1.若为震荡,则不进行操作
        2.若为下跌趋势,谨慎买入,尽量在低点买入,结合RSI,CCI,MACD_SIGNAL等信号判断低点,前两者应根据前30/40周期历史数据来判断,MACD_SINAL判断斜率和历史值,若无法判断则不进行操作
        3.若为上涨趋势,应稳健持有,卖出信号不应受到短期波动的影响(加密货币的波动是剧烈的),买入后适当调高风险阈值？或者减少风险判断,相信前面的买入操作,仅进行固定止损?
        """
        details = {
            'price': current_price,
            'timestamp': datetime.now()
        }

        if len(df) < 20:
            return "HOLD", "K线数据不足", details

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 在这里，生成技术信号之前，首先判断趋势：

        # 填充 details
        for col in ['macd', 'macd_signal', 'macd_slope', 'signal_slope', 'cci']:
            details[col] = latest.get(col, np.nan)

        # MACD 金叉/死叉
        macd_diff = latest['macd'] - latest['macd_signal']
        prev_macd_diff = prev['macd'] - prev['macd_signal']
        if prev_macd_diff <= 0 and macd_diff > 0 and abs(macd_diff) > self.config.MACD_CROSS_THRESHOLD:
            if self._confirm_signal(symbol, "BUY", df):
                return "BUY", f"MACD金叉确认: {macd_diff:.4f}", details

        if prev_macd_diff >= 0 and macd_diff < 0 and abs(macd_diff) > self.config.MACD_CROSS_THRESHOLD:
            if self._confirm_signal(symbol, "SELL", df):
                return "SELL", f"MACD死叉确认: {macd_diff:.4f}", details
        
        
        # MACD 斜率转正
        macd_slope = latest.get('macd_slope', 0)
        signal_slope = latest.get('signal_slope', 0)
        print(f"macd斜率和信号斜率: {macd_slope}, {signal_slope}")
        prev_macd_slope = prev.get('macd_slope', 0)
        macd_value = latest.get('macd', 0)
        signal_value = latest.get('macd_signal', 0)
        # 加入macd与signal差值判断震荡
        if prev_macd_slope < 0 and macd_slope > Config.MACD_POSITIVE_SLOPE_THRESHOLD and abs(macd_value - signal_value) > self.config.MACD_SIGNAL_DIFF_THRESHOLD:
            return "BUY", f"MACD斜率强势转正: {macd_slope:.4f}", details

        # MACD 在零轴上方斜率转负(<-2)
        if latest['macd'] > 0 and prev_macd_slope > 0 and macd_slope < -self.config.SIDEWAYS_SLOPE_THRESHOLD:
            return "SELL", f"MACD零轴上方斜率转负: {macd_slope:.4f}", details

        # CCI 超买超卖
        cci = latest.get('cci', 0)
        if cci > self.config.CCI_OVERBOUGHT:
            return "SELL", f"CCI超买: {cci:.2f}", details
        if cci < self.config.CCI_OVERSOLD:
            return "BUY", f"CCI超卖: {cci:.2f}", details

        # MACD 与信号线双向下
        # signal_slope = latest.get('signal_slope', 0)
        if macd_slope < -self.config.SIDEWAYS_SLOPE_THRESHOLD and signal_slope < -self.config.SIDEWAYS_SLOPE_THRESHOLD + 0.15: # +0.15是因为信号线更平稳
            return "SELL", f"双线向下加速: macd_slope={macd_slope:.4f}, signal_slope={signal_slope:.4f}", details

        # MACD 斜率强势上涨且快于信号线
        if macd_slope > Config.MACD_POSITIVE_SLOPE_THRESHOLD and macd_slope > signal_slope:
            return "BUY", f"MACD加速上涨: {macd_slope:.4f} > {signal_slope:.4f}", details

        # print(f"macd斜率和信号斜率：\n{macd_slope}\n{signal_slope}\n")
        return "HOLD", "技术指标无明确方向", details

    def _technical_signal_new(self, symbol: str, df: pd.DataFrame, current_price: float) -> Tuple[str, str, Dict]:
        """
        基于ADX和DMI判断趋势后生成技术信号
        """
        details = {
            'price': current_price,
            'timestamp': datetime.now()
        }

        if len(df) < 40:
            return "HOLD", "K线数据不足", details

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 填充 details
        for col in ['macd', 'macd_signal', 'macd_slope', 'signal_slope', 'cci', 'rsi']:
            details[col] = latest.get(col, np.nan)

        # 假设df已有adx, plus_di, minus_di（从calculate_adx_dmi计算）
        adx = latest.get('adx', 0)
        plus_di = latest.get('plus_di', 0)
        minus_di = latest.get('minus_di', 0)
        details.update({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })

        # 趋势判断
        if adx < self.config.ADX_OSCILLATION_THRESHOLD:  # e.g., 23
            trend = "OSCILLATION"
            trend_reason = f"震荡趋势 (ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f})"
            return "HOLD", f"{trend_reason}: 无明确方向，不操作", details

        if plus_di > minus_di and adx >= self.config.ADX_TREND_THRESHOLD:  # e.g., 25
            trend = "UP"
            trend_reason = f"上涨趋势 (ADX={adx:.2f}, +DI={plus_di:.2f} > -DI={minus_di:.2f})"
        elif minus_di > plus_di and adx >= self.config.ADX_TREND_THRESHOLD:
            trend = "DOWN"
            trend_reason = f"下跌趋势 (ADX={adx:.2f}, +DI={plus_di:.2f} < -DI={minus_di:.2f})"
        else:
            trend = "WEAK"
            trend_reason = f"趋势弱 (ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f})"
            return "HOLD", f"{trend_reason}: 观望", details

        details['trend'] = trend

        # MACD 金叉/死叉（通用，但根据趋势调整）
        macd_diff = latest['macd'] - latest['macd_signal']
        prev_macd_diff = prev['macd'] - prev['macd_signal']
        macd_slope = latest.get('macd_slope', 0)
        signal_slope = latest.get('signal_slope', 0)
        prev_macd_slope = prev.get('macd_slope', 0)
        macd_value = latest.get('macd', 0)
        signal_value = latest.get('macd_signal', 0)
        cci = latest.get('cci', 0)

        if trend == "UP":
            # 上涨趋势：稳健持有，卖出信号需更严格（调高阈值，忽略短期波动）
            # 买入信号：MACD金叉、斜率转正、MACD加速上涨、CCI超卖
            if prev_macd_diff <= 0 and macd_diff > 0 and abs(macd_diff) > self.config.MACD_CROSS_THRESHOLD:
                if self._confirm_signal(symbol, "BUY", df):
                    return "BUY", f"{trend_reason}: MACD金叉确认 {macd_diff:.4f}", details

            if prev_macd_slope < 0 and macd_slope > self.config.MACD_POSITIVE_SLOPE_THRESHOLD and abs(macd_value - signal_value) > self.config.MACD_SIGNAL_DIFF_THRESHOLD:
                return "BUY", f"{trend_reason}: MACD斜率强势转正 {macd_slope:.4f}", details

            if macd_slope > self.config.MACD_POSITIVE_SLOPE_THRESHOLD and macd_slope > signal_slope:
                return "BUY", f"{trend_reason}: MACD加速上涨 {macd_slope:.4f} > {signal_slope:.4f}", details

            if cci < self.config.CCI_OVERSOLD:
                return "BUY", f"{trend_reason}: CCI超卖 {cci:.2f}", details

            # 卖出信号：严格，只在MACD死叉（阈值调高）、零轴上方斜率转负、CCI超买、双线向下（需确认）
            adjusted_cross_threshold = self.config.MACD_CROSS_THRESHOLD * 2.4  # 调高阈值
            adjusted_cci_overbought = self.config.CCI_OVERBOUGHT + 50
            adjusted_sideways_threshold = self.config.MACD_POSITIVE_SLOPE_THRESHOLD * 2.4 # 调整上方斜率转负阈值

            if prev_macd_diff >= 0 and macd_diff < 0 and abs(macd_diff) > adjusted_cross_threshold:
                if self._confirm_signal(symbol, "SELL", df):
                    return "SELL", f"{trend_reason}: MACD死叉确认（严格） {macd_diff:.4f}", details

            if latest['macd'] > 0 and prev_macd_slope > 0 and macd_slope < -adjusted_sideways_threshold:
                return "SELL", f"{trend_reason}: MACD零轴上方斜率转负（严格） {macd_slope:.4f}", details

            if cci > adjusted_cci_overbought:
                return "SELL", f"{trend_reason}: CCI超买（严格） {cci:.2f}", details

            if macd_slope < -adjusted_sideways_threshold and signal_slope < -adjusted_sideways_threshold + 0.15:
                return "SELL", f"{trend_reason}: 双线向下加速（严格） macd_slope={macd_slope:.4f}, signal_slope={signal_slope:.4f}", details

            return "HOLD", f"{trend_reason}: 上涨趋势稳健持有，无明确买入或卖出信号", details

        elif trend == "DOWN":
            # 下跌趋势：谨慎买入，只在低点（结合RSI、CCI、MACD信号历史判断）
            # 先检查低点条件
            hist_rsi = df['rsi'].iloc[-40:-1]  # 前39周期
            hist_cci = df['cci'].iloc[-40:-1]
            hist_macd_signal = df['macd_signal'].iloc[-40:-1]

            rsi_mean, rsi_std = hist_rsi.mean(), hist_rsi.std()
            cci_mean, cci_std = hist_cci.mean(), hist_cci.std()
            macd_signal_low_quantile = hist_macd_signal.quantile(0.1)  # 10%低位

            is_low_point = (
                (latest.get('rsi', 0) < self.config.RSI_THRESHOLD and latest.get('rsi', 0) < rsi_mean - rsi_std) or
                (cci < self.config.CCI_OVERSOLD and cci < cci_mean - 1.5 * cci_std) or
                (latest['macd_signal'] < macd_signal_low_quantile and signal_slope > 0)  # 斜率转正
            )

            if not is_low_point:
                return "HOLD", f"{trend_reason}: 下跌趋势无可靠低点，不做多", details

            # 买入信号：只在低点时触发MACD金叉、斜率转正、MACD加速上涨、CCI超卖
            if prev_macd_diff <= 0 and macd_diff > 0 and abs(macd_diff) > self.config.MACD_CROSS_THRESHOLD:
                if self._confirm_signal(symbol, "BUY", df):
                    return "BUY", f"{trend_reason}: 低点MACD金叉确认 {macd_diff:.4f}", details

            if prev_macd_slope < 0 and macd_slope > self.config.MACD_POSITIVE_SLOPE_THRESHOLD and abs(macd_value - signal_value) > self.config.MACD_SIGNAL_DIFF_THRESHOLD:
                return "BUY", f"{trend_reason}: 低点MACD斜率强势转正 {macd_slope:.4f}", details

            if macd_slope > self.config.MACD_POSITIVE_SLOPE_THRESHOLD and macd_slope > signal_slope:
                return "BUY", f"{trend_reason}: 低点MACD加速上涨 {macd_slope:.4f} > {signal_slope:.4f}", details

            if cci < self.config.CCI_OVERSOLD:
                return "BUY", f"{trend_reason}: 低点CCI超卖 {cci:.2f}", details

            # 卖出信号：正常触发MACD死叉、零轴上方斜率转负、CCI超买、双线向下
            if prev_macd_diff >= 0 and macd_diff < 0 and abs(macd_diff) > self.config.MACD_CROSS_THRESHOLD:
                if self._confirm_signal(symbol, "SELL", df):
                    return "SELL", f"{trend_reason}: MACD死叉确认 {macd_diff:.4f}", details

            if latest['macd'] > 0 and prev_macd_slope > 0 and macd_slope < -self.config.SIDEWAYS_SLOPE_THRESHOLD:
                return "SELL", f"{trend_reason}: MACD零轴上方斜率转负 {macd_slope:.4f}", details

            if cci > self.config.CCI_OVERBOUGHT:
                return "SELL", f"{trend_reason}: CCI超买 {cci:.2f}", details

            if macd_slope < -self.config.SIDEWAYS_SLOPE_THRESHOLD and signal_slope < -self.config.SIDEWAYS_SLOPE_THRESHOLD + 0.15:
                return "SELL", f"{trend_reason}: 双线向下加速 macd_slope={macd_slope:.4f}, signal_slope={signal_slope:.4f}", details

            return "HOLD", f"{trend_reason}: 下跌趋势谨慎观望", details

        return "HOLD", f"{trend_reason}: 无明确方向", details

    def _confirm_signal(self, symbol: str, signal_type: str, df: pd.DataFrame) -> bool:
        """连续K线确认（原有逻辑保留）"""
        if symbol not in self.signal_states:
            self.signal_states[symbol] = {'signal_count': 0, 'signal_type': None}

        state = self.signal_states[symbol]
        latest = df.iloc[-1]

        current_macd_diff = latest['macd'] - latest['macd_signal']

        if signal_type == "BUY" and current_macd_diff > 0:
            if state['signal_type'] == "BUY":
                state['signal_count'] += 1
            else:
                state['signal_type'] = "BUY"
                state['signal_count'] = 1
        elif signal_type == "SELL" and current_macd_diff < 0:
            if state['signal_type'] == "SELL":
                state['signal_count'] += 1
            else:
                state['signal_type'] = "SELL"
                state['signal_count'] = 1
        else:
            state['signal_count'] = 0

        if state['signal_count'] >= self.config.CONFIRMATION_BARS:
            state['signal_count'] = 0
            return True
        return False

    def generate_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Tuple[str, str, Dict]:
        """
        主信号函数：双重确认机制
        """
        details = {'price': current_price, 'timestamp': datetime.now()}

        # 1. 获取数据库信号
        db_signal, db_reason = self._read_latest_db_signals(10)
        details['db_signal'] = db_signal
        details['db_reason'] = db_reason

        # 2. 获取技术指标信号
        tech_signal, tech_reason, tech_details = self._technical_signal_new(symbol, df, current_price)
        details.update(tech_details)
        details['tech_signal'] = tech_signal
        details['tech_reason'] = tech_reason

        # 3. 最终信号融合逻辑
        if "SELL" in (db_signal, tech_signal):  # 任意一方是 SELL，就离场
            final_signal = "SELL"
            reason = f"离场信号触发: 数据库({db_signal}/{db_reason}), 技术({tech_signal}/{tech_reason})"

        elif db_signal == "BUY" or tech_signal == "BUY":  # 至少一方是 BUY
            final_signal = "BUY"
            reason = f"确认做多！数据库信号：{db_signal}，技术信号：{tech_signal}（{tech_reason}）"
            self.logger.info(f"🚀 {symbol} 触发做多信号")

        else:
            final_signal = "HOLD"
            reason = f"持有观望: 数据库({db_signal}/{db_reason}), 技术({tech_signal}/{tech_reason})"

        details['final_signal'] = final_signal
        details['final_reason'] = reason

        # self.logger.info(f"{symbol} 信号: {final_signal} | 原因: {reason}")
        return final_signal, reason, details

    def clear_signal_state(self, symbol: str):
        """清除状态（平仓后调用）"""
        if symbol in self.signal_states:
            del self.signal_states[symbol]
            self.logger.info(f"已清除 {symbol} 信号状态")