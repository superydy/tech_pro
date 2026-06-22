"""
XGBoost 训练脚本（与 lgbm_train.py 配对，用于集成）
流程与 lgbm_train.py 完全一致，只换模型。
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
import multiprocessing as mp

CONFIG = {
    'data_path': './data/train.csv',
    'output_dir': './model/xgb',
    'top_k': 5,
    'n_splits': 4,
    'gap_days': 5,
    'val_days': 20,
    'xgb_params': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.02,
        'max_depth': 7,
        'min_child_weight': 30,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.05,
        'reg_lambda': 0.1,
        'seed': 42,
        'n_jobs': 4,
        'tree_method': 'hist',
        'verbosity': 0,
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
    dates = pd.to_datetime(dates_series)
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.values
    weights = 0.5 ** (age_days / half_life_days)
    return weights


def portfolio_return(pred_scores, true_labels, top_k=5):
    idx = np.argsort(pred_scores)[::-1][:top_k]
    return float(np.mean(true_labels[idx]))


def time_series_cv(df, feature_cols):
    dates = sorted(df['日期'].unique())
    n = len(dates)
    val_size = CONFIG['val_days']
    gap = CONFIG['gap_days']
    fold_results = []

    for fold in range(CONFIG['n_splits']):
        val_end_idx = n - fold * val_size - 1
        val_start_idx = val_end_idx - val_size + 1
        if val_start_idx < 0:
            break

        val_dates = set(dates[val_start_idx: val_end_idx + 1])
        train_end_idx = val_start_idx - gap - 1
        if train_end_idx < 60:
            break

        train_dates = set(dates[:train_end_idx + 1])
        train_df = df[df['日期'].isin(train_dates)].dropna(subset=feature_cols).copy()
        val_df   = df[df['日期'].isin(val_dates)].dropna(subset=feature_cols).copy()

        print(f"\n  Fold {fold+1}: 训练{len(train_dates)}天 | 验证{len(val_dates)}天")
        print(f"    训练: {min(train_dates)} ~ {max(train_dates)}")
        print(f"    验证: {min(val_dates)} ~ {max(val_dates)}")

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_val   = val_df[feature_cols].values
        y_val   = val_df['label'].values
        w_train = time_decay_weights(train_df['日期'])

        params = {k: v for k, v in CONFIG['xgb_params'].items() if k != 'n_estimators'}
        model = xgb.XGBRegressor(
            **params,
            n_estimators=CONFIG['xgb_params']['n_estimators'],
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

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


def train_final_model(df, feature_cols, num_rounds=None):
    print("\n在全量数据上训练最终 XGBoost 模型...")
    df = df.dropna(subset=feature_cols).copy()
    X = df[feature_cols].values
    y = df['label'].values
    w = time_decay_weights(df['日期'])

    n_rounds = num_rounds or CONFIG['xgb_params']['n_estimators']
    print(f"  使用 {n_rounds} 轮")

    params = {k: v for k, v in CONFIG['xgb_params'].items() if k != 'n_estimators'}
    model = xgb.XGBRegressor(**params, n_estimators=n_rounds)
    model.fit(X, y, sample_weight=w, verbose=200)
    return model


def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 50)
    print("加载数据...")
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from lgbm_features import build_features

    raw = pd.read_csv(CONFIG['data_path'])
    print(f"原始数据: {raw.shape}")

    df = build_features(raw, n_jobs=4)
    print(f"特征构建完成: {df.shape}")

    feature_cols = get_feature_cols(df)
    print(f"特征数量: {len(feature_cols)}")

    print("\n" + "=" * 50)
    print("开始时间序列交叉验证...")
    fold_results = time_series_cv(df, feature_cols)

    cv_returns = [r['val_return'] for r in fold_results]
    print(f"\n{'='*50}")
    print("XGBoost 交叉验证结果：")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['val_return']*100:.3f}%")
    print(f"  平均: {np.mean(cv_returns)*100:.3f}%  |  标准差: {np.std(cv_returns)*100:.3f}%")

    # 近期加权平均迭代次数（Fold 1 最新权重最高）
    iters = [(r['fold'], r['best_iter']) for r in fold_results if r.get('best_iter')]
    if iters:
        n = len(iters)
        wts = [n - i for i in range(n)]
        num_rounds = int(round(sum(wt * it for wt, (_, it) in zip(wts, iters)) / sum(wts)))
    else:
        num_rounds = CONFIG['xgb_params']['n_estimators']

    final_model = train_final_model(df, feature_cols, num_rounds=num_rounds)

    joblib.dump(final_model, os.path.join(CONFIG['output_dir'], 'xgb_model.pkl'))
    joblib.dump(feature_cols, os.path.join(CONFIG['output_dir'], 'feature_cols.pkl'))
    with open(os.path.join(CONFIG['output_dir'], 'cv_score.txt'), 'w') as f:
        f.write(f"CV avg weekly return: {np.mean(cv_returns)*100:.4f}%\n")
        f.write(f"CV std: {np.std(cv_returns)*100:.4f}%\n")
        for r in fold_results:
            f.write(f"Fold {r['fold']}: {r['val_return']*100:.4f}%\n")

    print(f"\nXGBoost 模型已保存到 {CONFIG['output_dir']}/")
    print(f"CV平均周收益: {np.mean(cv_returns)*100:.3f}%")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
