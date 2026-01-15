import requests
import csv
import time
from datetime import datetime

SYMBOL = "ETH_USDT"
INTERVAL = "1m"
LIMIT_PER_REQUEST = 1000
MAX_TOTAL_CANDLES = 5000

OUTPUT_FILE = "eth_usdt_1m_5000.csv"

url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"

params = {
    "contract": SYMBOL,
    "interval": INTERVAL,
    "limit": LIMIT_PER_REQUEST
}

all_data = []           # 最终去重后的数据
seen_timestamps = set() # 用于快速判断是否已存在

print("开始获取 ETH_USDT 1分钟K线...\n")

while len(all_data) < MAX_TOTAL_CANDLES:
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"请求失败: {e}")
        break

    items = r.json()

    if not items:
        print("返回空数据，结束获取")
        break

    added_this_batch = 0

    # 临时存储本次批次（先收集再去重）
    for item in items:
        if isinstance(item, dict):
            ts = int(item["t"])
            o = float(item["o"])
            h = float(item["h"])
            l = float(item["l"])
            c = float(item["c"])
            v = float(item["v"])
        else:  # list
            ts = int(item[0])
            o = float(item[1])
            h = float(item[3])
            l = float(item[4])
            c = float(item[2])
            v = float(item[5])

        # 只添加没见过的 timestamp
        if ts not in seen_timestamps:
            seen_timestamps.add(ts)
            all_data.append([ts, o, h, l, c, v])
            added_this_batch += 1

    if added_this_batch == 0:
        print("本次没有新增数据，结束获取")
        break

    # 更新下一页参数（使用本次返回中最小的 ts）
    if items:
        if isinstance(items[0], dict):
            min_ts_this_batch = min(int(item["t"]) for item in items)
        else:
            min_ts_this_batch = min(int(item[0]) for item in items)
        
        params["to"] = min_ts_this_batch - 1

    current_total = len(all_data)
    print(f"已获取 {current_total:,} 条")

    if current_total >= MAX_TOTAL_CANDLES:
        print(f"已达到目标 {MAX_TOTAL_CANDLES} 条，停止获取")
        break

    if len(items) < LIMIT_PER_REQUEST:
        print("返回数据少于limit，视为已取到最早数据")
        break

    time.sleep(0.8)

# 按时间升序排序
all_data.sort(key=lambda x: x[0])

# 截取最多10000条（以防万一）
all_data = all_data[:MAX_TOTAL_CANDLES]

# 写入 CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
    writer.writerows(all_data)

print(f"\n完成！")
print(f"总共保存 {len(all_data):,} 条唯一数据")
if all_data:
    print(f"时间范围: {datetime.fromtimestamp(all_data[0][0])} → {datetime.fromtimestamp(all_data[-1][0])}")
print(f"文件已保存至: {OUTPUT_FILE}")