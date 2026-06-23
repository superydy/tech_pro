"""
特征工程模块：
  1. per_stock_features     - 每只股票自己的时序特征（均线、动量、波动率等）
  2. cross_section_features - 横截面特征（当天在300只里的相对排名/超额收益）
  3. market_features        - 市场情绪特征（整体涨跌比、市场波动率）
  4. sector_features        - 行业特征（板块动量、股票在板块内的相对排名）
"""

import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────
# 0. 行业分类（用公司名关键词匹配）
# ─────────────────────────────────────────────
_SECTOR_RULES = [
    ('银行',    ['银行', '农商', '城商', '浦发', '兴业', '民生', '招商银行', '沪农', '渝农']),
    ('证券',    ['证券', '券商', '资本', '建投', '中金公司', '银河', '国泰', '国联', '广发', '方正', '申万宏源']),
    ('保险',    ['保险', '人寿', '人保', '太平', '中国平安', '中国太保', '新华保险']),
    ('地产',    ['地产', '房产', '置业', '城建', '万科', '保利', '华侨城', '招商蛇口']),
    ('医药',    ['医药', '医疗', '生物', '制药', '药业', '医院', '健康', '疫苗', '同仁堂', '片仔癀',
                 '药明', '云南白药', '长春高新', '华润三九', '上海莱士', '爱尔眼科', '康龙', '爱美客',
                 '新产业', '百利天恒', '华大九天']),
    ('食品饮料', ['茅台', '啤酒', '白酒', '食品', '饮料', '乳业', '牛奶', '调味', '海天', '伊利', '蒙牛',
                 '汾酒', '今世缘', '洋河', '五粮液', '古井', '舍得', '口子', '迎驾', '泸州老窖',
                 '双汇', '金龙鱼']),
    ('汽车',    ['汽车', '客车', '整车', '零部件', '比亚迪', '上汽', '广汽', '赛力斯', '拓普', '三花']),
    ('电子',    ['电子', '半导体', '芯片', '集成电路', '显示', '光学', '传感', '士兰微', '兆易', '瑞芯',
                 '工业富联', '豪威', '中微', '华勤', '传音', '澜起', '卓胜微', '龙芯', '盛美', '沪硅',
                 '华润微', '中芯', '京东方', '圣邦', '歌尔', '东山精密', '沪电', '立讯', '领益',
                 '深南电路', '鹏鼎', '三环', '中际旭创', '新易盛', '安克创新']),
    ('计算机',  ['软件', '信息', '科技', '网络', '数据', '用友', '东方财富', '恒生', '三六零', '中科曙光',
                 '宝信', '金山', '汉得', '广联达', '海康', '大华', '奇安信', '科大讯飞', '同花顺', '指南针',
                 '紫光股份', '昆仑万维']),
    ('通信',    ['通信', '联通', '移动', '电信', '中兴', '中国通号', '卫通', '烽火']),
    ('电力',    ['电力', '水电', '核电', '风电', '光伏', '电网', '国电', '华电', '华能', '浙能',
                 '特变电工', '东方电气', '明阳', '三峡', '长江电', '川投能源', '思源电气', '时代电气',
                 '中国广核']),
    ('新能源',  ['通威', '隆基', '阳光电源', '宁德', '亿纬', '新奥', '德业', '固德威',
                 '晶科能源', '大全能源', '阿特斯', '国轩高科', '天赐材料', '宝丰能源',
                 '晶盛机电']),
    ('化工',    ['化工', '化学', '石化', '化纤', '树脂', '轮胎', '农药', '巨化', '华鲁', '合盛', '万华',
                 '东方盛虹', '新和成']),
    ('钢铁',    ['钢铁', '钢股', '钢业', '宝钢', '包钢', '鞍钢', '华菱', '方大', '中信特钢']),
    ('有色金属', ['铜业', '铝业', '黄金', '稀土', '锂', '钴', '镍', '锌', '紫金', '洛阳钼', '赣锋', '天齐',
                 '铜陵有色', '云铝', '藏格', '盐湖', '龙佰', '山金', '光启']),
    ('煤炭',    ['煤炭', '煤业', '兖矿', '陕西煤', '中煤', '中国神华', '淮北矿', '开滦', '山西焦煤']),
    ('石油',    ['石油', '中国海油', '中国石油', '中国石化', '中海油服', '海油工程']),
    ('建材',    ['水泥', '玻璃', '建材', '巨石', '海螺', '三棵树', '东方雨虹']),
    ('建筑',    ['建筑', '路桥', '中建', '中铁', '中交', '中冶', '能建', '铁建', '交建', '招商公路']),
    ('机械设备', ['重工', '机械', '液压', '机床', '装备', '三一', '恒立', '中联', '振华', '汇川']),
    ('军工',    ['航空', '航天', '船舶', '军工', '兵器', '中航', '中船', '中车', '动力', '北方', '国货航', '紫光国微']),
    ('交通运输', ['机场', '港口', '高速', '航运', '物流', '快递', '铁路', '海运', '上港', '宁波港',
                 '青岛港', '中远', '招商轮船', '圆通', '申通', '顺丰', '韵达', '东航', '国航', '南航', '京沪']),
    ('农业',    ['农业', '农牧', '种业', '畜牧', '水产', '林业', '牧原', '海大集团', '温氏', '新希望']),
    ('家电',    ['家电', '电器', '空调', '冰箱', '洗衣', '美的', '格力', '海尔', '公牛', '苏泊尔']),
    ('纺织服装', ['纺织', '服装', '服饰', '制衣', '华利']),
    ('传媒',    ['传媒', '媒体', '影视', '广告', '出版', '文化', '芒果超媒']),
    ('商业零售', ['百货', '零售', '超市', '商场', '中免', '免税', '小商品']),
]

def build_sector_map(stock_list_path='./data/hs300_stock_list.csv') -> dict:
    """
    返回 {股票代码(6位字符串): 行业名称} 的字典。
    用公司名关键词匹配，匹配不到的归入"其他"。
    """
    sl = pd.read_csv(stock_list_path)
    sl['pure_code'] = sl['code'].str.replace(r'^[a-z]+\.', '', regex=True).str.zfill(6)

    sector_map = {}
    for _, row in sl.iterrows():
        code = row['pure_code']
        name = str(row['code_name'])
        matched = '其他'
        for sector, keywords in _SECTOR_RULES:
            if any(kw in name for kw in keywords):
                matched = sector
                break
        sector_map[code] = matched
    return sector_map


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
    ret1 = c.pct_change(1)

    # ── 收益率：短/中/长期动量 ──
    for w in [1, 3, 5, 10, 20, 40, 60, 120]:
        d[f'ret_{w}'] = c.pct_change(w)

    # ── 动量加速度（短期动量 vs 中长期动量之差，捕捉动量转折）──
    d['mom_accel_s'] = d['ret_5'] - d['ret_20']   # 短期加速
    d['mom_accel_m'] = d['ret_20'] - d['ret_60']  # 中期加速

    # ── 均线比率（价格偏离均线幅度，去掉绝对均线值）──
    for w in [5, 20, 40, 60, 120]:
        sma = c.rolling(w).mean()
        d[f'sma_ratio_{w}'] = c / (sma + 1e-9) - 1

    # ── 波动率 ──
    for w in [5, 10, 20]:
        d[f'vol_{w}'] = ret1.rolling(w).std()
    d['vol_ratio'] = d['vol_5'] / (d['vol_20'] + 1e-9)  # 波动率扩张/收缩

    # ── 成交量 ──
    for w in [5, 10, 20]:
        d[f'vma_{w}'] = v.rolling(w).mean()
    d['vratio_5'] = v / (d['vma_5'] + 1e-9)   # 短期量比
    d['vol_trend'] = d['vma_5'] / (d['vma_20'] + 1e-9)  # 量能趋势

    # ── MACD ──
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']

    # ── RSI ──
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── 布林带 ──
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d['boll_upper_dist'] = (c - (mid + 2 * std)) / (std + 1e-9)
    d['boll_lower_dist'] = (c - (mid - 2 * std)) / (std + 1e-9)
    d['boll_width'] = 4 * std / (mid + 1e-9)

    # ── 价格形态 ──
    d['high_low_ratio'] = (h - l) / (l + 1e-9)
    d['open_close_ratio'] = (c - o) / (o + 1e-9)
    d['upper_shadow'] = (h - c.clip(lower=o)) / (h - l + 1e-9)
    d['lower_shadow'] = (c.clip(upper=o) - l) / (h - l + 1e-9)

    # ── 换手率 ──
    t = d['换手率']
    for w in [5, 10, 20]:
        d[f'turn_ma_{w}'] = t.rolling(w).mean()
    d['turn_trend'] = d['turn_ma_5'] / (d['turn_ma_20'] + 1e-9)  # 换手率趋势

    # ── N日价格区间位置（52周位置是经典因子）──
    for w in [20, 60, 252]:
        hi = h.rolling(w).max()
        lo = l.rolling(w).min()
        d[f'close_pos_{w}'] = (c - lo) / (hi - lo + 1e-9)

    # ── 收益率连续性：近N日上涨天数占比 ──
    d['ret_pos_5'] = (ret1 > 0).rolling(5).mean()
    d['ret_pos_20'] = (ret1 > 0).rolling(20).mean()

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
    # 横截面排名：短/中/长期动量 + 波动率 + 换手率 + 动量加速度
    rank_cols = (
        [f'ret_{w}' for w in [1, 3, 5, 10, 20, 40, 60, 120]]
        + [f'vol_{w}' for w in [5, 10, 20]]
        + ['换手率', 'vratio_5', 'rsi_14', 'mom_accel_s', 'mom_accel_m', 'vol_ratio']
    )

    for col in rank_cols:
        if col not in df.columns:
            continue
        df[f'cs_rank_{col}'] = df.groupby('日期')[col].rank(pct=True)
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
# 4. 行业板块特征
# ─────────────────────────────────────────────
def sector_features(df: pd.DataFrame, stock_list_path: str = './data/hs300_stock_list.csv') -> pd.DataFrame:
    """
    行业特征：
    - sector_label       : 行业编号（用于模型学习行业效应）
    - sector_ret_N       : 同行业当天平均涨跌幅（板块动量）
    - sector_rank_ret_N  : 股票在本行业内的收益率排名（0~1）
    - sector_excess_ret_N: 股票收益率 - 本行业平均（是否跑赢行业）
    - sector_momentum_N  : 行业过去N日平均涨跌幅（行业中期趋势）
    """
    if not os.path.exists(stock_list_path):
        return df

    sector_map = build_sector_map(stock_list_path)
    # 股票代码统一为字符串6位
    df = df.copy()
    df['stock_code_str'] = df['股票代码'].astype(str).str.zfill(6)
    df['sector'] = df['stock_code_str'].map(sector_map).fillna('其他')

    # 行业编号（让模型能感知行业差异）
    sector_list = sorted(df['sector'].unique())
    sector2id = {s: i for i, s in enumerate(sector_list)}
    df['sector_label'] = df['sector'].map(sector2id).astype(int)

    # 按日期+行业计算板块统计
    for ret_col, windows in [('涨跌幅', [1, 5, 10]), ('ret_5', [5]), ('ret_20', [20])]:
        if ret_col not in df.columns:
            continue
        grp = df.groupby(['日期', 'sector'])[ret_col]
        col_suffix = ret_col.replace('涨跌幅', 'pct').replace('ret_', 'ret')

        # 当天行业平均
        df[f'sector_avg_{col_suffix}'] = grp.transform('mean')
        # 股票在行业内排名（0~1）
        df[f'sector_rank_{col_suffix}'] = grp.rank(pct=True)
        # 超额收益（跑赢/跑输行业多少）
        df[f'sector_excess_{col_suffix}'] = df[ret_col] - df[f'sector_avg_{col_suffix}']

    # 行业动量：行业日均涨跌幅的滚动均值（需要按行业排序后rolling）
    daily_sector = df.groupby(['日期', 'sector'])['涨跌幅'].mean().reset_index()
    daily_sector = daily_sector.sort_values(['sector', '日期'])
    for w in [5, 10, 20]:
        daily_sector[f'sector_momentum_{w}'] = (
            daily_sector.groupby('sector')['涨跌幅'].transform(lambda x: x.rolling(w).mean())
        )
    df = df.merge(
        daily_sector[['日期', 'sector'] + [f'sector_momentum_{w}' for w in [5, 10, 20]]],
        on=['日期', 'sector'], how='left'
    )

    df = df.drop(columns=['stock_code_str', 'sector'])
    return df


# ─────────────────────────────────────────────
# 5. 构建标签
# ─────────────────────────────────────────────
def build_label(df: pd.DataFrame, excess: bool = False) -> pd.DataFrame:
    """
    标签：T+1开盘到T+5开盘的收益率。
    excess=True：减去当日市场均值，变成超额收益（让模型学"谁比大盘强"）。
    excess=False（默认）：绝对收益（与比赛评分一致）。
    """
    df = df.copy()
    df = df.sort_values(['股票代码', '日期'])
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df = df.dropna(subset=['label', 'open_t1'])
    df = df[df['open_t1'] > 1e-4]
    df = df.drop(columns=['open_t1', 'open_t5'])

    if excess:
        # 减去当天所有股票的平均绝对收益 → 超额收益
        mkt_avg = df.groupby('日期')['label'].transform('mean')
        df['label'] = df['label'] - mkt_avg

    return df


# ─────────────────────────────────────────────
# 5. 完整特征流水线
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, n_jobs: int = 4, with_label: bool = True,
                   excess_label: bool = False) -> pd.DataFrame:
    """
    完整流水线：原始数据 → 所有特征（+ 标签）
    with_label=False 用于预测模式，保留最新行不丢弃
    excess_label=True 用超额收益作标签（学习相对alpha）
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

    # Step 4: 行业板块特征
    print("Step4: 计算行业特征...")
    df = sector_features(df)

    if with_label:
        # Step 5: 标签
        print("Step5: 构建标签...")
        df = build_label(df, excess=excess_label)

    # 清理
    df = df.replace([np.inf, -np.inf], np.nan)

    return df
