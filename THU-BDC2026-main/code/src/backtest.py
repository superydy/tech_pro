"""
稳健回测：长周期 walk-forward 评估

目的：解决 4折CV方差过大、无法区分模型优劣的问题。
做法：
  - 在最近 N 个月里，每隔 step 天设一个评估点
  - 每个点用截止当日的历史训练，预测未来 T+1→T+5 的 top-5 组合收益
  - 汇总几十~上百个周度样本，得到方差更小的业绩估计
  - 同时报告：平均收益、胜率、收益标准差、夏普

这才是接近比赛"提交一次→看一周收益"的真实模拟。
"""
import sys, os
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from lgbm_features import build_features
from lgbm_train import get_feature_cols, CONFIG, time_decay_weights, portfolio_return

# 回测配置
BACKTEST = {
    'eval_months': 18,      # 评估最近多少个月
    'step_days': 5,         # 每隔几天设一个评估点（5=每周一次，互不重叠）
    'gap_days': 5,          # 训练截止与预测之间的隔离
    'min_train_days': 500,  # 最少训练天数
    'top_k': 5,
    'retrain_every': 4,     # 每隔几个评估点重训一次模型（省时间）
}


def build_label_horizon(df):
    """T+1开盘 → T+5开盘 收益，作为每个交易日的前瞻标签"""
    df = df.sort_values(['股票代码', '日期']).copy()
    df['o1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['o5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['fwd_ret'] = (df['o5'] - df['o1']) / (df['o1'] + 1e-12)
    return df


_FEATURE_CACHE = {}


def _prepare_data():
    """构建特征 + 前瞻收益，只算一次并缓存（多配置对比时大幅省时）。"""
    if 'df' in _FEATURE_CACHE:
        return _FEATURE_CACHE['df'], _FEATURE_CACHE['fc'], _FEATURE_CACHE['fwd_map']

    raw = pd.read_csv(CONFIG['data_path'])
    df = build_features(raw, n_jobs=4)          # 含 label
    fc = get_feature_cols(df)

    # 真实前瞻收益（T+1开盘→T+5开盘），用未删尾的版本对齐
    full = build_features(raw, n_jobs=4, with_label=False)
    full = build_label_horizon(full)
    fwd_map = full.set_index(['日期', '股票代码'])['fwd_ret']

    _FEATURE_CACHE.update(df=df, fc=fc, fwd_map=fwd_map)
    return df, fc, fwd_map


def run_backtest(seeds=(42,), n_rounds=67):
    df, fc, fwd_map = _prepare_data()

    dates = sorted(df['日期'].unique())
    last_date = dates[-1]
    start_cut = last_date - pd.Timedelta(days=BACKTEST['eval_months'] * 30)
    eval_dates = [d for d in dates if d >= start_cut]
    eval_points = eval_dates[::BACKTEST['step_days']]

    print(f"回测区间: {eval_points[0].date()} ~ {eval_points[-1].date()}, 共 {len(eval_points)} 个评估点")

    results = []
    model_cache = None
    for i, ep in enumerate(eval_points):
        # 训练集：ep 之前留 gap 天
        tr_end_idx = dates.index(ep) - BACKTEST['gap_days']
        if tr_end_idx < BACKTEST['min_train_days']:
            continue
        tr_dates = set(dates[:tr_end_idx + 1])

        # 每 retrain_every 个点重训一次
        if model_cache is None or i % BACKTEST['retrain_every'] == 0:
            tr = df[df['日期'].isin(tr_dates)].dropna(subset=fc)
            Xtr, ytr = tr[fc].values, tr['label'].values
            wtr = time_decay_weights(tr['日期'])
            models = []
            for s in seeds:
                p = dict(CONFIG['lgbm_params']); p['random_state'] = s
                m = lgb.train(p, lgb.Dataset(Xtr, label=ytr, weight=wtr),
                              num_boost_round=n_rounds)
                models.append(m)
            model_cache = models

        # 预测当日
        day = df[df['日期'] == ep].dropna(subset=fc)
        if len(day) < BACKTEST['top_k']:
            continue
        Xd = day[fc].values
        # 多种子排名平均
        rank_sum = np.zeros(len(day))
        for m in model_cache:
            rank_sum += pd.Series(m.predict(Xd)).rank().values
        day = day.copy()
        day['score'] = rank_sum

        top = day.nlargest(BACKTEST['top_k'], 'score')
        # 用"真实前瞻收益"算组合收益
        rets = []
        for _, r in top.iterrows():
            key = (ep, r['股票代码'])
            if key in fwd_map.index:
                rets.append(fwd_map.loc[key])
        if rets:
            # 同时记录当周的市场状态（用于择时分析）
            # 取 ep 之前 10 个交易日的市场数据
            recent_idx = dates.index(ep)
            recent_dates = set(dates[max(0, recent_idx-10):recent_idx])
            mkt = df[df['日期'].isin(recent_dates)]
            mkt_ret5  = mkt.groupby('日期')['涨跌幅'].mean().tail(5).mean()
            mkt_ret10 = mkt.groupby('日期')['涨跌幅'].mean().mean()
            mkt_std5  = mkt.groupby('日期')['涨跌幅'].mean().tail(5).std()
            mkt_breadth = (mkt[mkt['日期'].isin(set(dates[max(0,recent_idx-5):recent_idx]))]['涨跌幅'] > 0).mean()
            results.append({
                'date': ep,
                'ret': np.mean(rets),
                'mkt_ret5': mkt_ret5,
                'mkt_ret10': mkt_ret10,
                'mkt_std5': mkt_std5,
                'mkt_breadth': mkt_breadth,
            })

    return pd.DataFrame(results)


def report(name, df_or_arr):
    if isinstance(df_or_arr, pd.DataFrame):
        results = df_or_arr['ret'].values
    else:
        results = np.asarray(df_or_arr)
    if len(results) == 0:
        print(f"{name}: 无有效样本")
        return
    avg = results.mean()
    win = (results > 0).mean()
    std = results.std()
    sharpe = avg / (std + 1e-9)
    print(f"\n=== {name} ===")
    print(f"  样本数(周): {len(results)}")
    print(f"  平均周收益: {avg*100:.3f}%")
    print(f"  胜率:       {win*100:.1f}%")
    print(f"  周收益标准差: {std*100:.3f}%")
    print(f"  单位风险收益(类夏普): {sharpe:.3f}")
    print(f"  累计收益(简单相加): {results.sum()*100:.1f}%")


def analyze_timing(res_df):
    """分析哪些市场状态下模型表现差，找择时规则。"""
    import numpy as np
    print("\n=== 择时分析：各市场状态下的周收益 ===")

    # 1. 按近5日市场均涨跌幅分组
    bins_ret = [-999, -1.5, -0.5, 0.5, 1.5, 999]
    labels_ret = ['<-1.5%', '-1.5~-0.5%', '-0.5~0.5%', '0.5~1.5%', '>1.5%']
    res_df['mkt_ret5_bin'] = pd.cut(res_df['mkt_ret5'], bins=bins_ret, labels=labels_ret)
    g = res_df.groupby('mkt_ret5_bin', observed=True)['ret'].agg(['mean','count','std'])
    g['win_rate'] = res_df.groupby('mkt_ret5_bin', observed=True)['ret'].apply(lambda x: (x>0).mean())
    print("\n近5日市场均涨跌幅 → 组合周收益：")
    print(f"  {'区间':<14} {'均值%':>7} {'胜率%':>7} {'标准差%':>8} {'样本':>5}")
    for label, row in g.iterrows():
        wr = res_df[res_df['mkt_ret5_bin']==label]['ret']
        print(f"  {str(label):<14} {row['mean']*100:>6.2f} {(wr>0).mean()*100:>7.1f} {row['std']*100:>8.2f} {int(row['count']):>5}")

    # 2. 按市场波动率分组
    bins_vol = [0, 0.5, 1.0, 1.5, 999]
    labels_vol = ['<0.5%', '0.5~1.0%', '1.0~1.5%', '>1.5%']
    res_df['mkt_std5_bin'] = pd.cut(res_df['mkt_std5'], bins=bins_vol, labels=labels_vol)
    g2 = res_df.groupby('mkt_std5_bin', observed=True)['ret']
    print("\n近5日市场波动率（日涨跌幅标准差）→ 组合周收益：")
    print(f"  {'区间':<12} {'均值%':>7} {'胜率%':>7} {'样本':>5}")
    for label, grp in g2:
        print(f"  {str(label):<12} {grp.mean()*100:>6.2f} {(grp>0).mean()*100:>7.1f} {len(grp):>5}")

    # 3. 按市场涨跌比例分组
    bins_br = [0, 0.35, 0.45, 0.55, 0.65, 1.0]
    labels_br = ['<35%', '35~45%', '45~55%', '55~65%', '>65%']
    res_df['breadth_bin'] = pd.cut(res_df['mkt_breadth'], bins=bins_br, labels=labels_br)
    g3 = res_df.groupby('breadth_bin', observed=True)['ret']
    print("\n近5日市场上涨家数占比 → 组合周收益：")
    print(f"  {'区间':<12} {'均值%':>7} {'胜率%':>7} {'样本':>5}")
    for label, grp in g3:
        print(f"  {str(label):<12} {grp.mean()*100:>6.2f} {(grp>0).mean()*100:>7.1f} {len(grp):>5}")


if __name__ == '__main__':
    import multiprocessing, argparse
    multiprocessing.set_start_method('spawn', force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--rounds', type=int, default=67)
    ap.add_argument('--analyze', action='store_true', help='输出择时分析')
    args = ap.parse_args()

    seed_pool = [42, 7, 123, 2024, 99]
    seeds = tuple(seed_pool[:args.seeds])
    print(f"【{len(seeds)} 种子】seeds={seeds}")
    res = run_backtest(seeds=seeds, n_rounds=args.rounds)
    report(f"{len(seeds)}种子 LightGBM", res)
    if args.analyze:
        analyze_timing(res)
