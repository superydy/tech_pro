"""
特征工程模块：
  1. per_stock_features  - 每只股票自己的时序特征（均线、动量、波动率等）
  2. cross_section_features - 横截面特征（当天在300只里的相对排名/超额收益）
  3. market_features      - 市场情绪特征（整体涨跌比、市场波动率）
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. 每只股票的时序特征
# ─────────────────────────────────────────────
def per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票按日期排序的数据
    输出：加上时序特征列的DataFrame
    """
    d = df.copy()
    c = d['收盘']
    v = d['成交量']
    h = d['最高']
    l = d['最低']
    o = d['开盘']

    # 收益率
    for w in [1, 3, 5, 10, 20, 40]:
        d[f'ret_{w}'] = c.pct_change(w)

    # 简单移动均线
    for w in [5, 10, 20, 40, 60]:
        d[f'sma_{w}'] = c.rolling(w).mean()
        d[f'sma_ratio_{w}'] = c / d[f'sma_{w}'] - 1  # 价格偏离均线的幅度

    # 波动率
    ret1 = c.pct_change(1)
    for w in [5, 10, 20]:
        d[f'vol_{w}'] = ret1.rolling(w).std()

    # 成交量特征
    for w in [5, 10, 20]:
        d[f'vmа_{w}'] = v.rolling(w).mean()
        d[f'vratio_{w}'] = v / (d[f'vmа_{w}'] + 1e-9)  # 量比

    # MACD（快慢线之差，不用TA-Lib手动算）
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']

    # RSI（14日）
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # 布林带
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d['boll_upper_dist'] = (c - (mid + 2 * std)) / (std + 1e-9)
    d['boll_lower_dist'] = (c - (mid - 2 * std)) / (std + 1e-9)
    d['boll_width'] = 4 * std / (mid + 1e-9)

    # 价格形态
    d['high_low_ratio'] = (h - l) / (l + 1e-9)
    d['open_close_ratio'] = (c - o) / (o + 1e-9)
    d['upper_shadow'] = (h - c.clip(lower=o)) / (h - l + 1e-9)
    d['lower_shadow'] = (c.clip(upper=o) - l) / (h - l + 1e-9)

    # 换手率滚动
    t = d['换手率']
    for w in [5, 10, 20]:
        d[f'turn_ma_{w}'] = t.rolling(w).mean()
        d[f'turn_ratio_{w}'] = t / (d[f'turn_ma_{w}'] + 1e-9)

    # 涨跌幅原始值保留
    d['pct_chg'] = d['涨跌幅']

    # 最高最低价位置
    for w in [5, 10, 20]:
        d[f'highest_{w}'] = h.rolling(w).max()
        d[f'lowest_{w}'] = l.rolling(w).min()
        d[f'close_pos_{w}'] = (c - d[f'lowest_{w}']) / (d[f'highest_{w}'] - d[f'lowest_{w}'] + 1e-9)

    return d


# ─────────────────────────────────────────────
# 2. 横截面特征（需要所有股票同一天数据）
# ─────────────────────────────────────────────
def cross_section_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：已做完per_stock_features的全量DataFrame（所有股票所有日期）
    输出：加上横截面特征列的DataFrame

    核心思想：把每只股票的指标，转化为"在当天300只里排第几"的相对位置。
    这样模型能感知"今天谁比其他股票强"，而不只是看自身绝对值。
    """
    df = df.copy()
    ret_cols = [f'ret_{w}' for w in [1, 3, 5, 10, 20]]
    vol_cols = [f'vol_{w}' for w in [5, 10, 20]]
    extra_cols = ['换手率', 'vratio_5', 'rsi_14']

    for col in ret_cols + vol_cols + extra_cols:
        if col not in df.columns:
            continue
        # 当天排名百分位（0=最低，1=最高）
        df[f'cs_rank_{col}'] = df.groupby('日期')[col].rank(pct=True)
        # 超额值（减去当天市场均值）
        df[f'cs_excess_{col}'] = df[col] - df.groupby('日期')[col].transform('mean')

    return df


# ─────────────────────────────────────────────
# 3. 市场整体情绪特征（每天一个值，所有股票共享）
# ─────────────────────────────────────────────
def market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：全量DataFrame
    输出：加上市场情绪特征列的DataFrame
    """
    df = df.copy()

    daily = df.groupby('日期').agg(
        mkt_up_ratio=('涨跌幅', lambda x: (x > 0).mean()),      # 上涨家数占比
        mkt_avg_ret=('涨跌幅', 'mean'),                          # 市场平均涨跌幅
        mkt_vol_ret=('涨跌幅', 'std'),                           # 市场涨跌幅波动
        mkt_avg_turn=('换手率', 'mean'),                         # 市场平均换手率
        mkt_vol_turn=('换手率', 'std'),                          # 换手率分化程度
    ).reset_index()

    # 市场动量：过去5/10天市场平均涨跌幅
    daily = daily.sort_values('日期')
    for w in [3, 5, 10]:
        daily[f'mkt_momentum_{w}'] = daily['mkt_avg_ret'].rolling(w).mean()
        daily[f'mkt_vol_ma_{w}'] = daily['mkt_vol_ret'].rolling(w).mean()

    df = df.merge(daily, on='日期', how='left')
    return df


# ─────────────────────────────────────────────
# 4. 构建标签
# ─────────────────────────────────────────────
def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    标签：T+1开盘到T+5开盘的收益率
    （与比赛评分完全一致）
    """
    df = df.copy()
    df = df.sort_values(['股票代码', '日期'])
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df = df.dropna(subset=['label', 'open_t1'])
    df = df[df['open_t1'] > 1e-4]
    df = df.drop(columns=['open_t1', 'open_t5'])
    return df


# ─────────────────────────────────────────────
# 5. 完整特征流水线
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, n_jobs: int = 4) -> pd.DataFrame:
    """
    完整流水线：原始数据 → 所有特征 + 标签
    """
    from multiprocessing import Pool
    from tqdm import tqdm

    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # Step 1: 每只股票分别算时序特征（并行）
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    print(f"Step1: 计算时序特征，共{len(groups)}只股票...")
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(per_stock_features, groups), total=len(groups)))
    df = pd.concat(results).reset_index(drop=True)

    # Step 2: 横截面特征（需要全量数据）
    print("Step2: 计算横截面特征...")
    df = cross_section_features(df)

    # Step 3: 市场情绪特征
    print("Step3: 计算市场情绪特征...")
    df = market_features(df)

    # Step 4: 标签
    print("Step4: 构建标签...")
    df = build_label(df)

    # 清理
    df = df.replace([np.inf, -np.inf], np.nan)

    return df
