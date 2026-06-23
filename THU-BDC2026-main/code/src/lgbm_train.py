"""
LightGBM Ranker 训练脚本

流程：
  1. 加载数据，计算所有特征
  2. 时间序列交叉验证（4折，折间留5天间隔）
  3. 每折训练 LightGBM Ranker
  4. OOF预测结果汇总 → 评估真实选股收益
  5. 保存最终模型（在全量数据上重训）
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import json
import multiprocessing as mp
from datetime import datetime, timedelta

# ── 配置 ──────────────────────────────────────────
CONFIG = {
    'data_path': './data/train.csv',
    'output_dir': './model/lgbm',
    'top_k': 5,
    'n_splits': 4,
    'gap_days': 5,       # 训练集和验证集之间的间隔天数（防止数据泄露）
    'val_days': 20,      # 每折验证集天数
    # 最终模型固定轮数（由72周稳健回测扫描选出，见 hpsweep）
    'final_rounds': 500,
    'lgbm_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'min_child_samples': 30,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.05,
        'reg_lambda': 0.1,
        'min_split_gain': 0.01,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1,
    }
}

FEATURE_BLACKLIST = {
    '股票代码', '日期', 'label',
    '开盘', '收盘', '最高', '最低',
    '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
}


def get_feature_cols(df):
    return [c for c in df.columns if c not in FEATURE_BLACKLIST]


def time_decay_weights(dates_series, half_life_days=500):
    """
    时间衰减样本权重：越近的数据权重越高。
    half_life_days天前的样本权重减半。
    """
    dates = pd.to_datetime(dates_series)
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.values
    weights = 0.5 ** (age_days / half_life_days)
    return weights


def portfolio_return(pred_scores, true_labels, top_k=5):
    """
    给定预测分数和真实收益，模拟选Top-k等权组合的实际收益。
    这才是比赛真正打分的方式。
    """
    idx = np.argsort(pred_scores)[::-1][:top_k]
    weights = np.ones(top_k) / top_k
    return float(np.dot(weights, true_labels[idx]))


def time_series_cv(df, feature_cols):
    """
    时间序列交叉验证（Walk-forward）
    - 验证集从最新往前切，共n_splits折
    - 折间留gap_days个交易日隔离，防止标签泄露
    """
    dates = sorted(df['日期'].unique())
    n = len(dates)
    val_size = CONFIG['val_days']
    gap = CONFIG['gap_days']

    fold_results = []

    for fold in range(CONFIG['n_splits']):
        # 验证集：从末尾往前数
        val_end_idx = n - fold * val_size - 1
        val_start_idx = val_end_idx - val_size + 1
        if val_start_idx < 0:
            break

        val_dates = set(dates[val_start_idx: val_end_idx + 1])
        train_end_idx = val_start_idx - gap - 1
        if train_end_idx < 60:  # 至少需要60天训练
            break

        train_dates = set(dates[:train_end_idx + 1])

        train_df = df[df['日期'].isin(train_dates)].copy()
        val_df = df[df['日期'].isin(val_dates)].copy()

        print(f"\n  Fold {fold+1}: 训练{len(train_dates)}天 | 验证{len(val_dates)}天")
        print(f"    训练: {min(train_dates)} ~ {max(train_dates)}")
        print(f"    验证: {min(val_dates)} ~ {max(val_dates)}")

        # 删除特征里的NaN
        train_df = train_df.dropna(subset=feature_cols)
        val_df = val_df.dropna(subset=feature_cols)

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['label'].values

        w_train = time_decay_weights(train_df['日期'])

        dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # 上限给足，让早停决定实际轮数（CV仅用于观察，不决定最终轮数）
        model = lgb.train(
            CONFIG['lgbm_params'],
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        # 评估：真实组合收益率
        val_df = val_df.copy()
        val_df['pred'] = model.predict(X_val)

        fold_returns = []
        for date, grp in val_df.groupby('日期'):
            if len(grp) < CONFIG['top_k']:
                continue
            ret = portfolio_return(grp['pred'].values, grp['label'].values, CONFIG['top_k'])
            fold_returns.append(ret)

        avg_ret = np.mean(fold_returns) if fold_returns else 0
        print(f"    验证集平均周收益率: {avg_ret*100:.3f}%")

        fold_results.append({
            'fold': fold + 1,
            'model': model,
            'val_return': avg_ret,
            'best_iter': model.best_iteration,
        })

    return fold_results


def _to_rank_label(df, labels):
    """把连续收益率标签转成整数排名标签（LightGBM Ranker需要）"""
    result = np.zeros(len(labels), dtype=np.int32)
    idx = 0
    for _, grp in df.groupby('日期'):
        n = len(grp)
        ranks = pd.Series(labels[idx:idx+n]).rank(method='first').astype(int).values
        # 归一到 0~4 的整数（5档相关性）
        result[idx:idx+n] = np.clip((ranks / n * 5).astype(int), 0, 4)
        idx += n
    return result


def train_final_model(df, feature_cols, num_rounds=None):
    """在全量数据上训练最终模型（用于提交）"""
    print("\n在全量数据上训练最终模型...")
    df = df.dropna(subset=feature_cols).copy()

    X = df[feature_cols].values
    y = df['label'].values
    w = time_decay_weights(df['日期'])

    n_rounds = num_rounds or CONFIG['final_rounds']
    print(f"  使用 {n_rounds} 轮（由72周稳健回测扫描选定）")

    dataset = lgb.Dataset(X, label=y, weight=w)
    model = lgb.train(
        CONFIG['lgbm_params'],
        dataset,
        num_boost_round=n_rounds,
        callbacks=[lgb.log_evaluation(200)],
    )
    return model


def main():
    mp.set_start_method('spawn', force=True)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # ── 1. 加载并构建特征 ──
    print("=" * 50)
    print("加载数据...")
    from lgbm_features import build_features
    raw = pd.read_csv(CONFIG['data_path'])
    print(f"原始数据: {raw.shape}")

    df = build_features(raw, n_jobs=4)
    print(f"特征构建完成: {df.shape}")

    feature_cols = get_feature_cols(df)
    print(f"特征数量: {len(feature_cols)}")

    # ── 2. 时间序列交叉验证 ──
    print("\n" + "=" * 50)
    print("开始时间序列交叉验证...")
    fold_results = time_series_cv(df, feature_cols)

    cv_returns = [r['val_return'] for r in fold_results]
    print(f"\n{'='*50}")
    print(f"交叉验证结果：")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['val_return']*100:.3f}%")
    print(f"  平均: {np.mean(cv_returns)*100:.3f}%  |  标准差: {np.std(cv_returns)*100:.3f}%")

    # ── 3. 训练最终模型 ──
    # 轮数固定为 CONFIG['final_rounds']（由72周稳健回测扫描选出，
    # 比CV早停的轮数更可靠——CV的组合收益指标方差太大）。
    final_model = train_final_model(df, feature_cols, num_rounds=CONFIG['final_rounds'])

    # ── 4. 保存 ──
    joblib.dump(final_model, os.path.join(CONFIG['output_dir'], 'lgbm_model.pkl'))
    joblib.dump(feature_cols, os.path.join(CONFIG['output_dir'], 'feature_cols.pkl'))
    with open(os.path.join(CONFIG['output_dir'], 'config.json'), 'w') as f:
        json.dump({**CONFIG, 'lgbm_params': {k: v for k, v in CONFIG['lgbm_params'].items()}}, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(CONFIG['output_dir'], 'cv_score.txt'), 'w') as f:
        f.write(f"CV avg weekly return: {np.mean(cv_returns)*100:.4f}%\n")
        f.write(f"CV std: {np.std(cv_returns)*100:.4f}%\n")
        for r in fold_results:
            f.write(f"Fold {r['fold']}: {r['val_return']*100:.4f}%\n")

    print(f"\n模型已保存到 {CONFIG['output_dir']}/")
    print(f"CV平均周收益: {np.mean(cv_returns)*100:.3f}%")


if __name__ == '__main__':
    main()
