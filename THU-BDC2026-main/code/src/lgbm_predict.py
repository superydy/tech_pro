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
    weights = weights / weights.sum()
    weights = np.floor(weights * 10000) / 10000  # 截断到4位小数，确保总和<=1
    return weights


def market_regime(df, latest_date, top_k=TOP_K):
    """
    数据驱动择时（72周回测分析得出）：
    - 舒适低波动牛市(ret5>0.5% AND std5<0.7%)：alpha最低，缩至3只
    - 高波动(std5>1.0%)：模型最强，满仓
    - 其他：正常持仓（微跌市场胜率75%，不再减仓）
    """
    recent = df[df['日期'] < str(latest_date)[:10]].copy()
    recent_dates = sorted(recent['日期'].unique())[-10:]
    recent = recent[recent['日期'].isin(recent_dates)]
    daily_ret = recent.groupby('日期')['涨跌幅'].mean()
    mkt_ret5 = daily_ret.tail(5).mean()
    mkt_std5 = daily_ret.tail(5).std()

    if mkt_std5 > 1.0:
        print(f"高波动市场(近5日std={mkt_std5:.2f}%)，模型最强，持仓{top_k}只")
        return top_k
    if mkt_ret5 > 0.5 and mkt_std5 < 0.7:
        n = max(3, top_k - 2)
        print(f"低波动微涨(ret5={mkt_ret5:.2f}%, std5={mkt_std5:.2f}%)，alpha偏低，缩仓至{n}只")
        return n
    print(f"市场正常(ret5={mkt_ret5:.2f}%, std5={mkt_std5:.2f}%)，持仓{top_k}只")
    return top_k


def main():
    os.makedirs('./output', exist_ok=True)

    # ── 1. 加载模型和特征列 ──
    model = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    print(f"模型加载完成，使用 {len(feature_cols)} 个特征")

    # ── 2. 加载数据，计算特征 ──
    print("加载数据并计算特征...")
    raw = pd.read_csv(DATA_PATH)
    # with_label=False：保留最新行，不丢弃没有T+5数据的行
    df = build_features(raw, n_jobs=4, with_label=False)
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
    n_hold = market_regime(raw, latest_date)

    # ── 6. 取Top-N，动态权重 ──
    top_df = today_df.nlargest(n_hold, 'pred_score').reset_index(drop=True)
    weights = dynamic_weights(top_df['pred_score'].values, n_hold)

    # ── 7. 生成 result.csv ──
    result_rows = []
    for i, row in top_df.iterrows():
        result_rows.append({
            'stock_id': str(row['股票代码']).zfill(6),
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
