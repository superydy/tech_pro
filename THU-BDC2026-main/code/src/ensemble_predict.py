"""
集成预测脚本：LightGBM + XGBoost 排名平均

排名平均（rank averaging）比直接平均分数更稳健：
- 两个模型的分数尺度不同，直接平均可能被一个模型主导
- 转换为日内排名后平均，每个模型贡献等权
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lgbm_features import build_features

LGB_DIR   = './model/lgbm'
XGB_DIR   = './model/xgb'
DATA_PATH = './data/train.csv'
OUTPUT_PATH = './output/result.csv'
TOP_K = 5


def dynamic_weights(scores, top_k):
    top_scores = scores[:top_k]
    exp_s = np.exp((top_scores - top_scores.max()) * 2)
    weights = exp_s / exp_s.sum()
    weights = np.clip(weights, 0.05, 0.40)
    weights = weights / weights.sum()
    weights = np.floor(weights * 10000) / 10000
    return weights


def market_regime(df, latest_date, top_k=TOP_K):
    """
    数据驱动择时（72周回测分析得出）：
    - 舒适低波动牛市(ret5>0.5% AND std5<0.7%)：alpha最低，缩至3只
    - 高波动(std5>1.0%)：模型最强，满仓
    - 其他：正常持仓
    注：微跌市场胜率75%，不再减仓
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

    # ── 1. 加载模型 ──
    lgb_model    = joblib.load(os.path.join(LGB_DIR, 'lgbm_model.pkl'))
    lgb_features = joblib.load(os.path.join(LGB_DIR, 'feature_cols.pkl'))
    print(f"LightGBM: {len(lgb_features)} 个特征")

    xgb_model    = None
    xgb_features = None
    xgb_path = os.path.join(XGB_DIR, 'xgb_model.pkl')
    if os.path.exists(xgb_path):
        xgb_model    = joblib.load(xgb_path)
        xgb_features = joblib.load(os.path.join(XGB_DIR, 'feature_cols.pkl'))
        print(f"XGBoost:   {len(xgb_features)} 个特征")
    else:
        print("XGBoost 模型未找到，退化为纯 LightGBM 预测")

    # ── 2. 特征构建 ──
    print("加载数据并计算特征...")
    raw = pd.read_csv(DATA_PATH)
    df  = build_features(raw, n_jobs=4, with_label=False)
    df  = df.replace([float('inf'), float('-inf')], float('nan'))

    latest_date = df['日期'].max()
    print(f"预测日期: {latest_date}")

    today_df = df[df['日期'] == latest_date].copy()

    # ── 3. 各模型打分 ──
    scores_list = []

    # LightGBM
    lgb_today = today_df.dropna(subset=lgb_features)
    X_lgb = lgb_today[lgb_features].values
    lgb_scores = lgb_model.predict(X_lgb)
    lgb_today = lgb_today.copy()
    lgb_today['lgb_score'] = lgb_scores
    # 转排名（越高越好 → rank从大到小，最高=1）
    lgb_today['lgb_rank'] = lgb_today['lgb_score'].rank(ascending=False)
    scores_list.append(lgb_today[['股票代码', 'lgb_rank']].rename(columns={'lgb_rank': 'rank_lgb'}))
    print(f"LightGBM 可预测: {len(lgb_today)} 只")

    # XGBoost
    if xgb_model is not None:
        xgb_today = today_df.dropna(subset=xgb_features)
        X_xgb = xgb_today[xgb_features].values
        xgb_scores = xgb_model.predict(X_xgb)
        xgb_today = xgb_today.copy()
        xgb_today['xgb_score'] = xgb_scores
        xgb_today['xgb_rank']  = xgb_today['xgb_score'].rank(ascending=False)
        scores_list.append(xgb_today[['股票代码', 'xgb_rank']].rename(columns={'xgb_rank': 'rank_xgb'}))
        print(f"XGBoost  可预测: {len(xgb_today)} 只")

    # ── 4. 排名平均合并 ──
    merged = scores_list[0]
    for s in scores_list[1:]:
        merged = merged.merge(s, on='股票代码', how='inner')

    rank_cols = [c for c in merged.columns if c.startswith('rank_')]
    merged['avg_rank'] = merged[rank_cols].mean(axis=1)

    # ── 5. 市场状态 → 持仓数量 ──
    n_hold = market_regime(raw, latest_date)

    # ── 6. 选 Top-N，动态权重 ──
    top_df = merged.nsmallest(n_hold, 'avg_rank').reset_index(drop=True)

    # 用 LightGBM 的原始分数来做 softmax 权重（排名只用于排序）
    top_codes = top_df['股票代码'].values
    top_lgb   = lgb_today[lgb_today['股票代码'].isin(top_codes)].set_index('股票代码')
    ordered_scores = np.array([top_lgb.loc[c, 'lgb_score'] for c in top_codes])
    weights = dynamic_weights(ordered_scores, n_hold)

    # ── 7. 输出 ──
    result_rows = [
        {'stock_id': str(code).zfill(6), 'weight': round(weights[i], 4)}
        for i, code in enumerate(top_codes)
    ]
    result = pd.DataFrame(result_rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*40}")
    used_models = ' + '.join(c.replace('rank_', '') for c in rank_cols)
    print(f"集成方式: {used_models} 排名平均")
    print("预测结果:")
    for _, r in result.iterrows():
        print(f"  {r['stock_id']}  权重: {r['weight']:.4f}")
    print(f"权重合计: {result['weight'].sum():.4f}")
    print(f"结果已写入: {OUTPUT_PATH}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
