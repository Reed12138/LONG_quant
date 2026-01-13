"""
回测框架（Backtester）
- 使用1m级别历史数据（CSV），按分钟推进回测
- 将1m数据实时聚合为30m K 线（可按需修改周期）
- 在每个1m步进时，用当前价格更新未完成的30m K线并计算指标
- 使用现有的 SignalGenerator（去掉数据库依赖）产生信号
- 使用虚拟账户模拟仓位、保证金、手续费、滑点与盈亏

使用示例:
    python backtester.py --symbol ETH_USDT --csv data/ETH_USDT_1m.csv --initial 10000

注意:
- 该回测尽量复用项目中已有的指标计算逻辑（DataFetcher）和信号逻辑（SignalGenerator）
- 回测中禁用数据库信号（_read_latest_db_signals 被覆盖为返回 HOLD）
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from config import Config
from data_fetcher import DataFetcher
from generate_signal import SignalGenerator
from utils import setup_logger


logger = setup_logger("Backtester", Config.LOG_LEVEL)


class BacktestSignalGenerator(SignalGenerator):
    """继承自 SignalGenerator，但禁用数据库访问（回测用）"""
    def __init__(self, config: Config):
        super().__init__(config)
        # 覆盖 db_config 防止误连接
        self.db_config = None

    def _read_latest_db_signals(self, limit: int = 10):
        # 回测时不依赖外部数据库，返回中性 HOLD
        return "HOLD", "backtest-disabled"


@dataclass
class Position:
    size: int
    entry_price: float
    margin: float
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class VirtualAccount:
    """简化的虚拟账户与仓位管理（兼顾杠杆和保证金）"""
    def __init__(self, initial_balance: float, config: Config):
        self.initial_balance = float(initial_balance)
        self.available = float(initial_balance)  # 可用资金
        self.used_margin = 0.0  # 被占用保证金
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.config = config
        self.trade_history: List[Dict] = []

    def _notional(self, size: int, price: float, contract_size: float) -> float:
        return abs(size) * contract_size * price

    def equity(self, current_price_lookup: Dict[str, float], contract_size: float = 0.01) -> float:
        unreal = 0.0
        for sym, pos in self.positions.items():
            if sym not in current_price_lookup:
                continue
            price = current_price_lookup[sym]
            unreal += (price - pos.entry_price) * pos.size * contract_size
        return self.available + self.used_margin + unreal

    def open_position(self, symbol: str, size: int, price: float, time: datetime, contract_size: float = 0.01) -> bool:
        """尝试开仓（市价成交）
        size: 正数为多，负数为空
        """
        if size == 0:
            return False

        # 如果已有同向仓位，叠加（简化）
        # 如果已有反向仓位，先全部平仓（保持和主程序一致）
        existing = self.positions.get(symbol)
        if existing and np.sign(existing.size) != np.sign(size):
            # 先平仓
            self.close_position(symbol, existing.size, price, time, contract_size)
            existing = None

        notional = self._notional(size, price, contract_size)
        required_margin = notional / self.config.LEVERAGE
        fee = notional * (self.config.HANDING_FEE_PCT / 100.0)

        if required_margin > self.available:
            logger.warning(f"资金不足无法开仓: 需要保证金 {required_margin:.2f}，可用 {self.available:.2f}")
            return False

        # 扣除保证金和手续费
        self.available -= required_margin
        self.used_margin += required_margin
        self.available -= fee  # 手续费从可用资金扣除

        if existing:
            # 加仓：计算新的加权均价
            prev_size = existing.size
            prev_entry = existing.entry_price
            new_size = prev_size + size
            if new_size == 0:
                # 刚好对冲平仓
                realized = (price - prev_entry) * prev_size * contract_size
                self.available += realized
                self.used_margin -= existing.margin
                self.trade_history.append({'time': time, 'symbol': symbol, 'size': size, 'price': price, 'realized': realized})
                del self.positions[symbol]
                return True
            # 更新 position
            # 简单按仓位加权平均入场价
            weighted_entry = (prev_entry * prev_size + price * size) / new_size
            existing.entry_price = weighted_entry
            existing.size = new_size
            existing.margin += required_margin
            existing.stop_loss = None
            existing.take_profit = None

            self.trade_history.append({'time': time, 'symbol': symbol, 'size': size, 'price': price, 'fee': fee, 'action': 'add'})
            return True

        # 新仓
        pos = Position(size=size, entry_price=price, margin=required_margin, open_time=time)
        # 设置止损/止盈（按 config 百分比）
        if size > 0:
            pos.stop_loss = price * (1.0 - self.config.STOP_LOSS_PCT / 100.0)
            pos.take_profit = price * (1.0 + self.config.TAKE_PROFIT_PCT / 100.0)
        else:
            pos.stop_loss = price * (1.0 + self.config.STOP_LOSS_PCT / 100.0)
            pos.take_profit = price * (1.0 - self.config.TAKE_PROFIT_PCT / 100.0)

        self.positions[symbol] = pos
        self.trade_history.append({'time': time, 'symbol': symbol, 'size': size, 'price': price, 'fee': fee, 'action': 'open'})
        logger.info(f"开仓: {symbol} {size} @ {price:.2f} (margin {required_margin:.2f}, fee {fee:.4f})")
        return True

    def close_position(self, symbol: str, size: int, price: float, time: datetime, contract_size: float = 0.01) -> bool:
        """平仓全部或部分（本实现仅支持全部平仓）"""
        pos = self.positions.get(symbol)
        if not pos:
            logger.warning(f"无仓位，无法平仓: {symbol}")
            return False

        # 仅支持按现有仓位全部平仓或全部反向（size 等于 pos.size）
        if size != pos.size:
            logger.warning("目前仅支持全部平仓的简化实现")
            size = pos.size

        notional = self._notional(pos.size, price, contract_size)
        fee = notional * (self.config.HANDING_FEE_PCT / 100.0)

        realized = (price - pos.entry_price) * pos.size * contract_size

        # 释放保证金
        self.used_margin -= pos.margin
        self.available += pos.margin

        # 加入已实现盈亏和扣手续费
        self.available += realized
        self.available -= fee

        self.trade_history.append({'time': time, 'symbol': symbol, 'size': -size, 'price': price, 'realized': realized, 'fee': fee, 'action': 'close'})
        logger.info(f"平仓: {symbol} {size} @ {price:.2f} realized {realized:.4f} fee {fee:.4f}")

        del self.positions[symbol]
        return True

    def check_stop_take(self, symbol: str, price: float, time: datetime, contract_size: float = 0.01):
        pos = self.positions.get(symbol)
        if not pos:
            return None
        # 检查止损/止盈成交
        if pos.size > 0:
            if price <= pos.stop_loss:
                self.close_position(symbol, pos.size, price, time, contract_size)
                return 'stop_loss'
            if price >= pos.take_profit:
                self.close_position(symbol, pos.size, price, time, contract_size)
                return 'take_profit'
        else:
            if price >= pos.stop_loss:
                self.close_position(symbol, pos.size, price, time, contract_size)
                return 'stop_loss'
            if price <= pos.take_profit:
                self.close_position(symbol, pos.size, price, time, contract_size)
                return 'take_profit'
        return None


class Backtester:
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = DataFetcher(config)
        self.signal_gen = BacktestSignalGenerator(config)
        self.account = None
        self.contract_size = 0.01  # 默认合约面值，与你实盘一致或从合约配置文件读取
        self.equity_curve: List[Dict] = []

    @staticmethod
    def load_1m_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # 期望 CSV 包含 timestamp, open, high, low, close, volume
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            df.set_index('timestamp', inplace=True)
        else:
            raise ValueError('CSV 需要包含 timestamp 或 time 列')
        df = df.sort_index()
        return df[['open', 'high', 'low', 'close', 'volume']]

    def run(self, symbol: str, df_1m: pd.DataFrame, initial_balance: float = 10000.0):
        """主回测循环：按1m步进，动态维护30m K线并生成信号、成交"""
        self.account = VirtualAccount(initial_balance, self.config)

        # 保证 1m 数据按时间升序
        df_1m = df_1m.sort_index()

        # 用于维护 30m K 线
        k_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        last_bar_start = None
        price_lookup = {symbol: None}

        for ts, row in df_1m.iterrows():
            price = float(row['close'])
            vol = float(row.get('volume', 0.0))
            price_lookup[symbol] = price

            bar_start = ts.floor('30min')

            # 新周期
            if last_bar_start is None or bar_start > last_bar_start:
                # 新建一根30m未完成K线
                new_row = pd.Series({'open': price, 'high': price, 'low': price, 'close': price, 'volume': vol}, name=bar_start)
                k_df = pd.concat([k_df, new_row.to_frame().T])
                last_bar_start = bar_start
            else:
                # 更新当前未完成K线
                idx = k_df.index[-1]
                k_df.loc[idx, 'close'] = price
                k_df.loc[idx, 'high'] = max(k_df.loc[idx, 'high'], price)
                k_df.loc[idx, 'low'] = min(k_df.loc[idx, 'low'], price)
                k_df.loc[idx, 'volume'] = k_df.loc[idx, 'volume'] + vol

            # 只有当有足够30m K线数时才计算指标
            if len(k_df) >= max(self.config.LOOKBACK_PERIODS, 50):
                # 复制并计算指标（DataFetcher 的计算函数在内存中运行）
                df_30 = k_df.copy()
                df_30 = self.fetcher.calculate_macd(df_30)
                df_30 = self.fetcher.calculate_cci(df_30)
                df_30 = self.fetcher.calculate_volume_ma(df_30)
                df_30 = self.fetcher.calculate_adx_dmi_safe(df_30)
                df_30 = self.fetcher.calculate_rsi(df_30)

                # 将最新收盘价替换为当前价格（逼近实时）
                df_30.loc[df_30.index[-1], 'close'] = price

                # 生成信号
                signal, reason, details = self.signal_gen.generate_signal(symbol, df_30, price)

                # 执行信号（简化逻辑，参照主程序）
                now = ts.to_pydatetime()
                if signal == 'CLEAR':
                    # 平掉所有仓位
                    if symbol in self.account.positions:
                        pos = self.account.positions[symbol]
                        self.account.close_position(symbol, pos.size, price, now, self.contract_size)
                elif signal == 'BUY':
                    # 若持空仓，先平空，再开多
                    existing = self.account.positions.get(symbol)
                    if existing and existing.size < 0:
                        self.account.close_position(symbol, existing.size, price, now, self.contract_size)
                        self.account.open_position(symbol, int(self.config.SIZE), price, now, self.contract_size)
                    elif not existing:
                        self.account.open_position(symbol, int(self.config.SIZE), price, now, self.contract_size)
                elif signal == 'SELL':
                    existing = self.account.positions.get(symbol)
                    if existing and existing.size > 0:
                        self.account.close_position(symbol, existing.size, price, now, self.contract_size)
                        self.account.open_position(symbol, -int(self.config.SIZE), price, now, self.contract_size)
                    elif not existing:
                        self.account.open_position(symbol, -int(self.config.SIZE), price, now, self.contract_size)

                # 检查止损/止盈
                c = self.account.check_stop_take(symbol, price, now, self.contract_size)
                if c:
                    logger.info(f"{symbol} 在 {ts} 触发 {c}")

            # 记录权益曲线（按分钟记录）
            eq = self.account.equity({symbol: price}, self.contract_size)
            self.equity_curve.append({'time': ts, 'equity': eq})

        # 回测结束，关闭所有仓位（按最后价格）
        last_price = price_lookup[symbol]
        if symbol in self.account.positions:
            pos = self.account.positions[symbol]
            self.account.close_position(symbol, pos.size, last_price, df_1m.index[-1].to_pydatetime(), self.contract_size)

        # 生成报告
        return self.report()

    def report(self) -> Dict:
        df = pd.DataFrame(self.equity_curve).set_index('time')
        df = df.sort_index()

        initial = self.initial_capital() if self.account else 0
        final = df['equity'].iloc[-1]
        total_return = (final / initial - 1.0) * 100.0

        # 计算日收益率用于夏普（近似）
        daily = df['equity'].resample('1D').last().pct_change().dropna()
        sharpe = None
        if not daily.empty and daily.std() > 0:
            sharpe = (daily.mean() / daily.std()) * np.sqrt(252)

        # 最大回撤
        cummax = df['equity'].cummax()
        drawdown = (df['equity'] - cummax) / cummax
        max_dd = drawdown.min() * 100.0

        trades = pd.DataFrame(self.account.trade_history)
        wins = trades[trades['action'] == 'close']
        win_rate = None
        if not wins.empty:
            win_rate = (wins['realized'] > 0).mean()

        report = {
            'initial': initial,
            'final': final,
            'total_return_pct': total_return,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd,
            'n_trades': len(trades),
            'win_rate': win_rate,
            'trades': trades
        }

        # 打印摘要
        logger.info("\n回测结果摘要")
        logger.info(f"初始资金: {initial:.2f} -> 结束: {final:.2f} | 收益率: {total_return:.2f}%")
        logger.info(f"交易次数: {len(trades)} | 胜率(平仓): {win_rate}")
        logger.info(f"最大回撤: {max_dd:.2f}% | 夏普: {sharpe}")

        return report

    def initial_capital(self) -> float:
        return self.account.initial_balance if self.account else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--initial', type=float, default=10000.0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bt = Backtester(Config())
    df1 = Backtester.load_1m_csv(args.csv)
    report = bt.run(args.symbol, df1, initial_balance=args.initial)

    # 保存交易记录和报告
    if 'trades' in report and not report['trades'].empty:
        report['trades'].to_csv('backtest_trades.csv', index=False)
    import json
    with open('backtest_report.json', 'w', encoding='utf-8') as f:
        json.dump({k: (v.tolist() if isinstance(v, pd.Series) else (v if not hasattr(v, 'to_dict') else v.to_dict())) for k, v in report.items() if k != 'trades'}, f, ensure_ascii=False, indent=2)

    logger.info('回测完成，交易记录保存在 backtest_trades.csv，报告保存在 backtest_report.json')
