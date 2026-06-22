"""
LightGBM 预测脚本

流程：
  1. 加载全量历史数据，计算特征
  2. 取最新交易日，对300只股票打分
  3. 动态权重：按预测分数高低分配，高分股票多投
  4. 加市场状态过滤：市场整体较弱时，减少持仓数量
  5. 输出 result.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lgbm_features import build_features

MODEL_DIR = './model/lgbm'
DATA_PATH = './data/train.csv'
OUTPUT_PATH = './output/result.csv'
TOP_K = 5


def dynamic_weights(scores, top_k):
    """
    按预测分数高低分配权重：
    分数越高投越多，但总和=1，每只最多0.4。
    比固定0.2更聪明。
    """
    top_scores = scores[:top_k]
    # softmax使高分股票权重更大
    exp_s = np.exp((top_scores - top_scores.max()) * 2)
    weights = exp_s / exp_s.sum()
    weights = np.clip(weights, 0.05, 0.40)
    weights = weights / weights.sum()  # 归一化确保加起来=1
    return weights


def market_ok(df, latest_date):
    """
    判断市场状态：
    - 最近5天市场平均涨跌幅 < -2%: 市场较弱，减少持仓
    - 否则正常
    返回建议持仓数量
    """
    recent = df[df['日期'] < str(latest_date)[:10]].copy()
    recent_dates = sorted(recent['日期'].unique())[-5:]
    recent = recent[recent['日期'].isin(recent_dates)]
    avg_ret = recent.groupby('日期')['涨跌幅'].mean().mean()
    mkt_vol = recent.groupby('日期')['涨跌幅'].std().mean()

    if avg_ret < -2.0:
        print(f"市场偏弱（近5日均涨跌幅{avg_ret:.2f}%），减少持仓到3只")
        return 3
    if avg_ret < -1.0 and mkt_vol > 3.0:
        print(f"市场震荡（近5日均涨跌幅{avg_ret:.2f}%，波动{mkt_vol:.2f}%），持仓4只")
        return 4
    print(f"市场状态正常（近5日均涨跌幅{avg_ret:.2f}%），持仓{TOP_K}只")
    return TOP_K


def main():
    os.makedirs('./output', exist_ok=True)

    # ── 1. 加载模型和特征列 ──
    model = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    print(f"模型加载完成，使用 {len(feature_cols)} 个特征")

    # ── 2. 加载数据，计算特征 ──
    print("加载数据并计算特征...")
    raw = pd.read_csv(DATA_PATH)
    df = build_features(raw, n_jobs=4)
    df = df.replace([float('inf'), float('-inf')], float('nan'))

    # ── 3. 取最新交易日 ──
    latest_date = df['日期'].max()
    print(f"预测日期: {latest_date}")

    today_df = df[df['日期'] == latest_date].copy()
    today_df = today_df.dropna(subset=feature_cols)
    print(f"当日可预测股票数: {len(today_df)}")

    if len(today_df) == 0:
        print("当日无可预测股票！")
        return

    # ── 4. 打分 ──
    X = today_df[feature_cols].values
    scores = model.predict(X)
    today_df = today_df.copy()
    today_df['pred_score'] = scores

    # ── 5. 市场状态判断 ──
    n_hold = market_ok(raw, latest_date)

    # ── 6. 取Top-N，动态权重 ──
    top_df = today_df.nlargest(n_hold, 'pred_score').reset_index(drop=True)
    weights = dynamic_weights(top_df['pred_score'].values, n_hold)

    # ── 7. 生成 result.csv ──
    # 股票代码还原为6位字符串
    # train.csv里股票代码是数字索引，需要对应回原始代码
    # 读stock_list拿真实代码
    stock_list = pd.read_csv('./data/hs300_stock_list.csv')
    stock_list['pure_code'] = stock_list['code'].str.replace(r'^[a-z]+\.', '', regex=True).str.zfill(6)

    # train.csv里的股票代码是排序后的索引（1,2,63...）
    # 先建立索引→真实代码的映射
    raw2 = pd.read_csv(DATA_PATH)
    raw2['股票代码'] = raw2['股票代码'].astype(str)
    all_codes = sorted(raw2['股票代码'].unique(), key=lambda x: int(x))
    # stock_list按代码排序
    sl_sorted = sorted(stock_list['pure_code'].tolist())

    # 映射：train里的索引 → 真实6位代码
    idx2code = {str(i + 1): code for i, code in enumerate(sl_sorted)}

    result_rows = []
    for i, row in top_df.iterrows():
        raw_code = str(int(row['股票代码']))
        real_code = idx2code.get(raw_code, raw_code.zfill(6))
        result_rows.append({
            'stock_id': real_code,
            'weight': round(weights[len(result_rows)], 4)
        })

    result = pd.DataFrame(result_rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*40}")
    print("预测结果:")
    for _, r in result.iterrows():
        print(f"  {r['stock_id']}  权重: {r['weight']:.4f}")
    print(f"权重合计: {result['weight'].sum():.4f}")
    print(f"结果已写入: {OUTPUT_PATH}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
