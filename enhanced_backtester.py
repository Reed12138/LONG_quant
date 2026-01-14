"""
增强版回测框架（Enhanced Backtester）
- 使用1m级别历史数据，动态聚合为30m K线
- 模拟实盘的实时K线更新逻辑（_append_or_update_live_bar）
- 完整的保证金、杠杆、手续费、滑点模拟
- 详细的性能指标和交易分析
- 支持移动止损和风险管理

使用示例:
    python enhanced_backtester.py --symbol ETH_USDT --csv data/eth_usdt_1m_10000.csv --initial 10000
"""

import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from config import Config
from data_fetcher import DataFetcher
from generate_signal import SignalGenerator
from risk_manager import RiskManager
from utils import setup_logger


logger = setup_logger("EnhancedBacktester", Config.LOG_LEVEL)


class BacktestSignalGenerator(SignalGenerator):
    """回测专用信号生成器（禁用数据库访问）"""
    def __init__(self, config: Config):
        super().__init__(config)
        self.db_config = None
        self._db_signal_override = "HOLD"  # 可在回测中设置模拟信号

    def _read_latest_db_signals(self, limit: int = 10):
        """回测时返回预设的数据库信号（默认HOLD）"""
        return self._db_signal_override, "backtest-simulated"
    
    def set_db_signal(self, signal: str, reason: str = "backtest"):
        """允许回测中模拟数据库信号变化"""
        self._db_signal_override = signal


@dataclass
class TradeRecord:
    """交易记录"""
    time: datetime
    symbol: str
    action: str  # 'open_long', 'open_short', 'close', 'stop_loss', 'take_profit'
    size: int
    price: float
    fee: float
    realized_pnl: float = 0.0
    balance_after: float = 0.0
    reason: str = ""


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    size: int  # 正数=多头，负数=空头
    entry_price: float
    entry_time: datetime
    margin: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_profit_pct: float = 0.0  # 最大浮盈百分比（用于移动止损）
    
    def unrealized_pnl(self, current_price: float, contract_size: float = 0.01) -> float:
        """计算未实现盈亏"""
        return (current_price - self.entry_price) * self.size * contract_size
    
    def unrealized_pnl_pct(self, current_price: float, contract_size: float = 0.01) -> float:
        """计算未实现盈亏百分比（基于保证金）"""
        if self.margin <= 0:
            return 0.0
        return self.unrealized_pnl(current_price, contract_size) / self.margin


class VirtualAccount:
    """虚拟账户（完整模拟实盘逻辑）"""
    def __init__(self, initial_balance: float, config: Config, contract_size: float = 0.01):
        self.initial_balance = float(initial_balance)
        self.available = float(initial_balance)
        self.used_margin = 0.0
        self.config = config
        self.contract_size = contract_size
        
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        
        # 统计数据
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
    
    def total_equity(self, price_lookup: Dict[str, float]) -> float:
        """计算总权益（可用资金 + 占用保证金 + 未实现盈亏）"""
        unrealized = sum(
            pos.unrealized_pnl(price_lookup.get(sym, pos.entry_price), self.contract_size)
            for sym, pos in self.positions.items()
        )
        return self.available + self.used_margin + unrealized
    
    def _calculate_fee(self, size: int, price: float) -> float:
        """计算手续费"""
        notional = abs(size) * self.contract_size * price
        return notional * (self.config.HANDING_FEE_PCT / 100.0)
    
    def _calculate_margin(self, size: int, price: float) -> float:
        """计算所需保证金"""
        notional = abs(size) * self.contract_size * price
        return notional / self.config.LEVERAGE
    
    def can_open_position(self, size: int, price: float) -> bool:
        """检查是否有足够资金开仓"""
        required_margin = self._calculate_margin(size, price)
        fee = self._calculate_fee(size, price)
        return (required_margin + fee) <= self.available
    
    def open_position(self, symbol: str, size: int, price: float, time: datetime, reason: str = "") -> bool:
        """开仓"""
        if size == 0:
            return False
        
        # 检查是否已有反向仓位，如有则先平仓
        existing = self.positions.get(symbol)
        if existing and np.sign(existing.size) != np.sign(size):
            self.close_position(symbol, price, time, "reverse_open")
        
        # 计算所需保证金和手续费
        required_margin = self._calculate_margin(size, price)
        fee = self._calculate_fee(size, price)
        
        if not self.can_open_position(size, price):
            logger.warning(f"资金不足: 需要 {required_margin + fee:.2f}, 可用 {self.available:.2f}")
            return False
        
        # 扣除保证金和手续费
        self.available -= required_margin + fee
        self.used_margin += required_margin
        self.total_fees += fee
        
        # 创建持仓
        pos = Position(
            symbol=symbol,
            size=size,
            entry_price=price,
            entry_time=time,
            margin=required_margin
        )
        
        # 设置止损止盈
        # if size > 0:  # 多头
        #     pos.stop_loss = price * (1.0 - self.config.STOP_LOSS_PCT / 100.0)
        #     pos.take_profit = price * (1.0 + self.config.TAKE_PROFIT_PCT / 100.0)
        # else:  # 空头
        #     pos.stop_loss = price * (1.0 + self.config.STOP_LOSS_PCT / 100.0)
        #     pos.take_profit = price * (1.0 - self.config.TAKE_PROFIT_PCT / 100.0)

        # 计算基于保证金的止盈止损（使用已有的 PCT 参数）
        notional_value = abs(size) * self.contract_size * price  # 名义仓位价值

        # 1. 止损：最大亏损金额 = required_margin × (STOP_LOSS_PCT / 100)
        risk_amount = required_margin * (self.config.STOP_LOSS_PCT / 100.0)
        price_risk_per_contract = risk_amount / (self.contract_size * abs(size))

        # 2. 止盈：目标盈利金额 = required_margin × (TAKE_PROFIT_PCT / 100)
        reward_amount = required_margin * (self.config.TAKE_PROFIT_PCT / 100.0)
        price_reward_per_contract = reward_amount / (self.contract_size * abs(size))

        # 3. 根据多空方向设置价格
        if size > 0:  # 多头
            pos.stop_loss   = price - price_risk_per_contract
            pos.take_profit = price + price_reward_per_contract
        else:  # 空头
            pos.stop_loss   = price + price_risk_per_contract    # 价格上涨 → 亏损
            pos.take_profit = price - price_reward_per_contract  # 价格下跌 → 盈利
        
        self.positions[symbol] = pos
        self.total_trades += 1
        
        action = 'open_long' if size > 0 else 'open_short'
        trade = TradeRecord(
            time=time,
            symbol=symbol,
            action=action,
            size=size,
            price=price,
            fee=fee,
            balance_after=self.total_equity({symbol: price}),
            reason=reason
        )
        self.trade_history.append(trade)
        
        logger.debug(f"{action}: {symbol} {abs(size)}张 @ {price:.2f}, 保证金 {required_margin:.2f}, 手续费 {fee:.4f}")
        return True
    
    def close_position(self, symbol: str, price: float, time: datetime, reason: str = "") -> bool:
        """平仓"""
        pos = self.positions.get(symbol)
        if not pos:
            return False
        
        # 计算盈亏和手续费
        realized_pnl = pos.unrealized_pnl(price, self.contract_size)
        fee = self._calculate_fee(pos.size, price)
        
        # 释放保证金，结算盈亏，扣手续费
        self.used_margin -= pos.margin
        self.available += pos.margin + realized_pnl - fee
        self.total_fees += fee
        
        # 统计胜负
        if realized_pnl > 0:
            self.winning_trades += 1
        elif realized_pnl < 0:
            self.losing_trades += 1
        
        trade = TradeRecord(
            time=time,
            symbol=symbol,
            action='close',
            size=-pos.size,
            price=price,
            fee=fee,
            realized_pnl=realized_pnl,
            balance_after=self.available + self.used_margin,
            reason=reason
        )
        self.trade_history.append(trade)
        
        logger.debug(f"平仓: {symbol} {abs(pos.size)}张 @ {price:.2f}, 盈亏 {realized_pnl:.4f}, 手续费 {fee:.4f}")
        
        del self.positions[symbol]
        return True
    
    def check_stop_loss_take_profit(self, symbol: str, price: float, time: datetime) -> Optional[str]:
        """检查止损止盈"""
        pos = self.positions.get(symbol)
        if not pos:
            return None
        
        if pos.size > 0:  # 多头
            if price <= pos.stop_loss:
                self.close_position(symbol, price, time, "stop_loss")
                logger.info(f"多头止损触发: {price:.2f}<{pos.stop_loss:.2f}")
                return 'stop_loss'
            if price >= pos.take_profit:
                self.close_position(symbol, price, time, "take_profit")
                logger.info(f"多头止盈触发: {price:.2f}>{pos.take_profit:.2f}")
                return 'take_profit'
        else:  # 空头
            if price >= pos.stop_loss:
                self.close_position(symbol, price, time, "stop_loss")
                logger.info(f"空头止损触发: {price:.2f}>{pos.stop_loss:.2f}")
                return 'stop_loss'
            if price <= pos.take_profit:
                self.close_position(symbol, price, time, "take_profit")
                logger.info(f"空头止盈触发: {price:.2f}<{pos.take_profit:.2f}")
                return 'take_profit'
        
        return None
    
    def check_trailing_stop(self, symbol: str, price: float, time: datetime) -> bool:
        """检查移动止损"""
        pos = self.positions.get(symbol)
        if not pos:
            return False
        
        pnl_pct = pos.unrealized_pnl_pct(price, self.contract_size) * 100
        
        # 更新最大浮盈
        if pnl_pct > pos.max_profit_pct:
            pos.max_profit_pct = pnl_pct
        
        peak = pos.max_profit_pct
        current = pnl_pct
        
        # 移动止损逻辑（与实盘一致）
        if peak >= self.config.TRAILING_STOP_PEAK:
            if current <= peak * 0.9:
                self.close_position(symbol, price, time, "trailing_stop_high")
                logger.info(f"移动止损（高位）: 峰值 {peak:.2f}%, 当前 {current:.2f}%")
                return True
        elif peak >= self.config.TRAILING_STOP_LOW:
            if current <= peak * 0.8:
                self.close_position(symbol, price, time, "trailing_stop_low")
                logger.info(f"移动止损（低位）: 峰值 {peak:.2f}%, 当前 {current:.2f}%")
                return True
        
        return False
    
    def record_equity(self, time: datetime, price_lookup: Dict[str, float]):
        """记录权益曲线"""
        equity = self.total_equity(price_lookup)
        self.equity_curve.append({
            'time': time,
            'equity': equity,
            'available': self.available,
            'used_margin': self.used_margin,
            'positions': len(self.positions)
        })


class EnhancedBacktester:
    """增强版回测器"""
    def __init__(self, config: Config, contract_size: float = 0.01):
        self.config = config
        self.contract_size = contract_size
        
        self.fetcher = DataFetcher(config)
        self.signal_gen = BacktestSignalGenerator(config)
        self.risk_manager = RiskManager(config)
        self.account: Optional[VirtualAccount] = None
        
        # 30m K线数据（动态维护）
        self.k30m_df = pd.DataFrame()
        self.last_bar_start = None
        
    @staticmethod
    def load_1m_csv(path: str) -> pd.DataFrame:
        """加载1分钟K线数据"""
        df = pd.read_csv(path)
        
        # 处理时间列
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={'time': 'timestamp'}, inplace=True)
        else:
            raise ValueError('CSV需要包含timestamp或time列')
        
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def _update_30m_bar(self, timestamp: pd.Timestamp, price: float, volume: float):
        """
        模拟实盘的 _append_or_update_live_bar 逻辑
        动态维护30分钟K线
        """
        bar_start = timestamp.floor('30min')
        
        # 新周期开始
        if self.last_bar_start is None or bar_start > self.last_bar_start:
            # 添加新的30m K线
            new_bar = pd.Series({
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }, name=bar_start)
            
            self.k30m_df = pd.concat([self.k30m_df, new_bar.to_frame().T])
            self.last_bar_start = bar_start
        else:
            # 更新当前未完成的K线
            idx = self.k30m_df.index[-1]
            self.k30m_df.loc[idx, 'close'] = price
            self.k30m_df.loc[idx, 'high'] = max(self.k30m_df.loc[idx, 'high'], price)
            self.k30m_df.loc[idx, 'low'] = min(self.k30m_df.loc[idx, 'low'], price)
            self.k30m_df.loc[idx, 'volume'] += volume
    
    def _calculate_indicators(self) -> pd.DataFrame:
        """计算所有技术指标"""
        if len(self.k30m_df) < 50:
            return self.k30m_df.copy()
        
        df = self.k30m_df.copy()
        
        # 计算指标（使用DataFetcher的方法）
        df = self.fetcher.calculate_macd(df)
        df = self.fetcher.calculate_cci(df)
        df = self.fetcher.calculate_volume_ma(df)
        df = self.fetcher.calculate_adx_dmi_safe(df)
        df = self.fetcher.calculate_rsi(df)
        df = self.fetcher.calculate_vei(df)
        
        return df
    
    def run(self, symbol: str, df_1m: pd.DataFrame, initial_balance: float = 10000.0) -> Dict:
        """
        执行回测
        
        Args:
            symbol: 交易对
            df_1m: 1分钟K线数据
            initial_balance: 初始资金
            
        Returns:
            回测报告
        """
        logger.info(f"开始回测: {symbol}")
        logger.info(f"数据范围: {df_1m.index[0]} 至 {df_1m.index[-1]} ({len(df_1m)} 根1m K线)")
        logger.info(f"初始资金: {initial_balance:.2f} USDT")
        
        self.account = VirtualAccount(initial_balance, self.config, self.contract_size)
        self.k30m_df = pd.DataFrame()
        self.last_bar_start = None
        
        # 确保数据按时间排序
        df_1m = df_1m.sort_index()
        
        # 逐分钟推进
        for timestamp, row in df_1m.iterrows():
            price = float(row['close'])
            volume = float(row['volume'])
            
            # 更新30m K线
            self._update_30m_bar(timestamp, price, volume)
            
            # 等待足够的30m K线数据
            if len(self.k30m_df) < max(50, self.config.LOOKBACK_PERIODS // 4):
                continue
            
            # 计算指标
            df_with_indicators = self._calculate_indicators()
            
            # 生成交易信号
            signal, reason, details = self.signal_gen.generate_signal(
                symbol, 
                df_with_indicators, 
                price
            )
            
            # 先检查风险（止损止盈、移动止损）
            if symbol in self.account.positions:
                # 检查移动止损
                if self.account.check_trailing_stop(symbol, price, timestamp):
                    continue
                
                # 检查止损止盈
                result = self.account.check_stop_loss_take_profit(symbol, price, timestamp)
                if result:
                    continue
            
            # 执行交易信号
            if signal == 'CLEAR':
                if symbol in self.account.positions:
                    self.account.close_position(symbol, price, timestamp, f"CLEAR: {reason}")
            
            elif signal == 'BUY':
                existing = self.account.positions.get(symbol)
                if existing:
                    if existing.size < 0:  # 持空仓，反向开多
                        self.account.close_position(symbol, price, timestamp, "reverse")
                        self.account.open_position(symbol, self.config.SIZE, price, timestamp, f"BUY: {reason}")
                else:  # 无仓位，开多
                    self.account.open_position(symbol, self.config.SIZE, price, timestamp, f"BUY: {reason}")
            
            elif signal == 'SELL':
                existing = self.account.positions.get(symbol)
                if existing:
                    if existing.size > 0:  # 持多仓，反向开空
                        self.account.close_position(symbol, price, timestamp, "reverse")
                        self.account.open_position(symbol, -self.config.SIZE, price, timestamp, f"SELL: {reason}")
                else:  # 无仓位，开空
                    self.account.open_position(symbol, -self.config.SIZE, price, timestamp, f"SELL: {reason}")
            
            # 记录权益曲线（每小时记录一次）
            if timestamp.minute == 0:
                self.account.record_equity(timestamp, {symbol: price})
        
        # 回测结束，平掉所有仓位
        final_price = df_1m.iloc[-1]['close']
        final_time = df_1m.index[-1]
        
        for sym in list(self.account.positions.keys()):
            self.account.close_position(sym, final_price, final_time, "backtest_end")
        
        # 最后记录一次权益
        self.account.record_equity(final_time, {symbol: final_price})
        
        # 生成报告
        return self.generate_report(symbol, df_1m.index[0], df_1m.index[-1])
    
    def generate_report(self, symbol: str, start_time, end_time) -> Dict:
        """生成回测报告"""
        logger.info("\n" + "="*60)
        logger.info("回测报告")
        logger.info("="*60)
        
        initial = self.account.initial_balance
        final = self.account.available + self.account.used_margin
        
        # 基础指标
        total_return = (final / initial - 1.0) * 100.0
        duration = (end_time - start_time).days
        
        logger.info(f"交易对: {symbol}")
        logger.info(f"回测周期: {start_time} 至 {end_time} ({duration}天)")
        logger.info(f"初始资金: {initial:.2f} USDT")
        logger.info(f"最终资金: {final:.2f} USDT")
        logger.info(f"总收益率: {total_return:.2f}%")
        
        # 交易统计
        trades_df = pd.DataFrame([t.__dict__ for t in self.account.trade_history])
        closed_trades = trades_df[trades_df['action'] == 'close']
        
        if not closed_trades.empty:
            win_rate = (closed_trades['realized_pnl'] > 0).mean() * 100
            avg_win = closed_trades[closed_trades['realized_pnl'] > 0]['realized_pnl'].mean()
            avg_loss = closed_trades[closed_trades['realized_pnl'] < 0]['realized_pnl'].mean()
            
            logger.info(f"\n交易次数: {len(closed_trades)}")
            logger.info(f"胜率: {win_rate:.2f}%")
            logger.info(f"盈利交易: {self.account.winning_trades}")
            logger.info(f"亏损交易: {self.account.losing_trades}")
            logger.info(f"平均盈利: {avg_win:.2f} USDT" if not np.isnan(avg_win) else "平均盈利: N/A")
            logger.info(f"平均亏损: {avg_loss:.2f} USDT" if not np.isnan(avg_loss) else "平均亏损: N/A")
            logger.info(f"总手续费: {self.account.total_fees:.2f} USDT")
        
        # 权益曲线分析
        if self.account.equity_curve:
            equity_df = pd.DataFrame(self.account.equity_curve).set_index('time')
            equity_series = equity_df['equity']
            
            # 最大回撤
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax
            max_drawdown = drawdown.min() * 100
            
            # 夏普比率（简化版）
            if len(equity_series) > 1:
                returns = equity_series.pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # 按小时计算
                else:
                    sharpe = 0
            else:
                sharpe = 0
            
            logger.info(f"\n最大回撤: {max_drawdown:.2f}%")
            logger.info(f"夏普比率: {sharpe:.2f}")
        
        logger.info("="*60 + "\n")
        
        # 返回详细报告
        return {
            'symbol': symbol,
            'start_time': str(start_time),
            'end_time': str(end_time),
            'duration_days': duration,
            'initial_balance': initial,
            'final_balance': final,
            'total_return_pct': total_return,
            'total_trades': len(closed_trades) if not closed_trades.empty else 0,
            'winning_trades': self.account.winning_trades,
            'losing_trades': self.account.losing_trades,
            'win_rate': win_rate if not closed_trades.empty else 0,
            'avg_win': avg_win if not closed_trades.empty and not np.isnan(avg_win) else 0,
            'avg_loss': avg_loss if not closed_trades.empty and not np.isnan(avg_loss) else 0,
            'total_fees': self.account.total_fees,
            'max_drawdown_pct': max_drawdown if self.account.equity_curve else 0,
            'sharpe_ratio': sharpe if self.account.equity_curve else 0,
            'trades': trades_df.to_dict('records') if not trades_df.empty else [],
            'equity_curve': self.account.equity_curve
        }
    
    def save_results(self, report: Dict, output_prefix: str = "backtest"):
        """保存回测结果"""
        # 保存交易记录
        if report['trades']:
            trades_df = pd.DataFrame(report['trades'])
            trades_file = f"{output_prefix}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"交易记录已保存: {trades_file}")
        
        # 保存权益曲线
        if report['equity_curve']:
            equity_df = pd.DataFrame(report['equity_curve'])
            equity_file = f"{output_prefix}_equity.csv"
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"权益曲线已保存: {equity_file}")
        
        # 保存JSON报告（排除大型数据）
        json_report = {k: v for k, v in report.items() if k not in ['trades', 'equity_curve']}
        report_file = f"{output_prefix}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"回测报告已保存: {report_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='加密货币策略回测工具')
    parser.add_argument('--symbol', type=str, default='ETH_USDT', help='交易对')
    parser.add_argument('--csv', type=str, required=True, help='1分钟K线数据CSV文件路径')
    parser.add_argument('--initial', type=float, default=10000.0, help='初始资金（USDT）')
    parser.add_argument('--output', type=str, default='backtest', help='输出文件前缀')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = Config()
    
    # 创建回测器
    backtester = EnhancedBacktester(config)
    
    # 加载数据
    logger.info(f"加载数据: {args.csv}")
    df_1m = EnhancedBacktester.load_1m_csv(args.csv)
    
    # 执行回测
    report = backtester.run(
        symbol=args.symbol,
        df_1m=df_1m,
        initial_balance=args.initial
    )
    
    # 保存结果
    backtester.save_results(report, args.output)
    
    logger.info("回测完成！")


if __name__ == '__main__':
    main()