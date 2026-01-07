"""
加密货币量化交易系统配置模块
所有可调参数集中在此模块，便于管理和优化
"""

class Config:
    """
    系统配置类
    包含所有可调整的参数和设置
    """
    
    # ============ Gate.io API 配置 ============
    API_BASE_URL = "https://api.gateio.ws/api/v4"
    
    # ============ 交易对配置 ============
    SYMBOLS = ["ETH_USDT"]  # 交易对列表, 仅交易ETH_USDT
    QUOTE_CURRENCY = "USDT"  # 计价货币
    
    # ============ 时间周期配置 ============
    INTERVAL = "30m"  # K线周期
    LOOKBACK_PERIODS = 200  # 历史数据回看周期数
    INTERVAL_SECONDS = 60  # 程序运行时间间隔
    SLEEP_INTERVAL = 300   # 程序休眠时间间隔
    SLEEP_INTERVAL_TRAILING_STOP = 900 # 触发移动止损后，程序休眠时间间隔

    # ============ MACD 指标参数 ============
    MACD_FAST_LENGTH = 6    # 快线周期（短期EMA）
    MACD_SLOW_LENGTH = 13   # 慢线周期（长期EMA）
    MACD_SIGNAL_LENGTH = 9  # 信号线周期
    
    # ============ CCI 指标参数 ============
    CCI_PERIOD = 20         # CCI计算周期
    
    # ============ 震荡参数 =============
    MACD_SIGNAL_DIFF_THRESHOLD = 0.8  # MACD信号差异阈值
    ADX_PERIOD = 14                   # ADX计算周期
    ADX_TREND_THRESHOLD = 18      # 趋势强度阈值
    ADX_OSCILLATION_THRESHOLD = 18  # 低于此值为震荡
    RSI_THRESHOLD = 44
    RSI_PERIOD = 14

    # ============ 交易信号参数 ============
    # MACD交叉信号阈值（避免微小波动触发）
    MACD_CROSS_THRESHOLD = 0.6
    
    # MACD正斜率下限值：
    MACD_POSITIVE_SLOPE_THRESHOLD = 0.32  # 正斜率阈值, 用于确认上涨趋势
    SIDEWAYS_SLOPE_THRESHOLD = 0.12        # 震荡斜率阈值

    # CCI超买超卖阈值
    CCI_OVERBOUGHT = 280    # 超买阈值
    CCI_OVERSOLD = -240     # 超卖阈值
    
    # 信号确认条件
    CONFIRMATION_BARS = 3   # 信号确认需要的K线数量
    
    # ============ 风险管理参数 ============
    # 止损止盈参数
    STOP_LOSS_PCT = 2.8     # 止损百分比 %
    TAKE_PROFIT_PCT = 8   # 止盈百分比 %
    TRAILING_STOP_PCT = 10  # 移动止损百分比 %
    HANDING_FEE_PCT = 2.5    # 手续费百分比 %
    
    # 移动止损参数
    TRAILING_STOP_PEAK = 20 # %
    TRAILING_STOP_LOW = 6 # %

    # 仓位管理 暂时不用
    POSITION_SIZE_PCT = 10.0  # 每笔交易仓位比例（占总资金%）
    MAX_POSITION_PCT = 60.0  # 最大总仓位比例
    
    # 换手率异常检测 暂时不用
    VOLUME_MA_PERIOD = 50    # 成交量均线周期
    VOLUME_SPIKE_THRESHOLD = 2.5  # 成交量异常阈值（超过均线的倍数）
    VOLUME_DROP_THRESHOLD = 0.3   # 成交量萎缩阈值（低于均线的比例）
    
    # MACD顶背离检测
    DIVERGENCE_LOOKBACK = 30  # 背离检测回看周期
    MIN_PEAK_DISTANCE = 5     # 峰值最小距离
    MACD_SLOPE_LOOKBACK = 3    # 斜率计算回看周期
    
    # ============ 交易执行参数 ============
    ORDER_TIMEOUT = 30  # 订单超时时间（秒）
    RETRY_ATTEMPTS = 3  # 重试次数
    
    # ============= 交易配置 =============
    # SIMULATION_MODE = False  # 是否为模拟交易模式
    # INITIAL_CAPITAL = 10000.0  # 初始资金（USDT）
    SETTLE = "usdt" # 合约的结算币种
    LEVERAGE = 5  # 杠杆倍数
    SIZE = 1 # 每次开仓的合约张数
    
    # ============ 日志和监控 ============
    LOG_LEVEL = "INFO"  # 日志级别：DEBUG, INFO, WARNING, ERROR
    SAVE_TRADES_TO_FILE = True  # 是否保存交易记录到文件
    PRINT_TRADE_SUMMARY = True  # 是否打印交易摘要