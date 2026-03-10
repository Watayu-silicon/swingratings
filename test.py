import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
from itertools import combinations
import json
import os

# =========================================================
# 共通テクニカル指標計算
# =========================================================
def calculate_wilders_rsi(price_series: pd.Series, period: int = 14) -> pd.Series:
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_bollinger_bands(price_series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = price_series.rolling(window=period).mean()
    std = price_series.rolling(window=period).std(ddof=0)
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return lower_band

# =========================================================
# A. 予想RSI (R_proj) 算出アルゴリズム
# =========================================================
def evaluate_tomorrow_bounce_rsi(df: pd.DataFrame, ma_col: str) -> dict:
    if len(df) < 15:
        return {"status": "error", "message": "データが不足しています"}

    current_price = df['Close'].iloc[-1]
    current_ma = df[ma_col].iloc[-1]
    distance_pct = (current_price - current_ma) / current_ma * 100

    simulated_close_touch = current_ma
    simulated_close_bounce = current_ma * 1.015
    close_series = df['Close'].reset_index(drop=True)

    series_touch = pd.concat([close_series, pd.Series([simulated_close_touch])], ignore_index=True)
    series_bounce = pd.concat([close_series, pd.Series([simulated_close_bounce])], ignore_index=True)

    rsi_touch = calculate_wilders_rsi(series_touch).iloc[-1]
    rsi_bounce = calculate_wilders_rsi(series_bounce).iloc[-1]
    projected_rsi_avg = (rsi_touch + rsi_bounce) / 2

    return {
        "status": "target",
        "current_distance_pct": round(distance_pct, 2),
        "projected_rsi_touch": round(rsi_touch, 2),
        "projected_rsi_bounce": round(rsi_bounce, 2),
        "projected_rsi": round(projected_rsi_avg, 2)
    }

# =========================================================
# B. 反発勝率とノック回数 算出アルゴリズム（プライスアクション確認・厳格版）
# =========================================================
def calculate_bounce_history(df: pd.DataFrame, ma_col: str, lookback_days: int = 380,
                             recent_knock_days: int = 30, touch_margin: float = 0.012,
                             win_margin: float = 0.05, win_window: int = 5) -> dict:

    df_target = df.tail(lookback_days).copy().reset_index(drop=True)
    if len(df_target) < win_window + 10:
        return {"status": "error", "p_bounce": 0.0, "k_knocks": 0, "total_events": 0, "win_events": 0}

    df_target['Next_Open'] = df_target['Open'].shift(-1)
    df_target['Next_Close'] = df_target['Close'].shift(-1)

    cond_a = (
        ((abs(df_target['Low'] - df_target[ma_col]) / df_target[ma_col]) <= touch_margin) |
        ((abs(df_target['Close'] - df_target[ma_col]) / df_target[ma_col]) <= touch_margin)
    )
    cond_b1 = df_target['Next_Open'] > df_target['Close']
    cond_b2 = df_target['Next_Close'] > df_target['Close']
    cond_c = df_target['Next_Close'] > df_target['Next_Open']

    df_target['is_knock'] = cond_a & cond_b1 & cond_b2 & cond_c
    knock_positions = df_target[df_target['is_knock']].index.tolist()

    if not knock_positions:
        return {"status": "no_touches", "p_bounce": 0.0, "k_knocks": 0, "total_events": 0, "win_events": 0}

    events = []
    last_pos = -999
    event_cooldown = 3

    for pos in knock_positions:
        if pos - last_pos > event_cooldown:
            events.append(pos)
        last_pos = pos

    win_count = 0
    recent_knocks = 0
    total_events = len(events)
    latest_pos = len(df_target) - 1

    for pos in events:
        target_high = df_target.loc[pos, ma_col] * (1.0 + win_margin)
        end_pos = min(pos + win_window + 1, len(df_target))
        if df_target.loc[pos:end_pos-1, 'High'].max() >= target_high:
            win_count += 1

        if latest_pos - pos <= recent_knock_days:
            recent_knocks += 1

    return {
        "status": "success",
        "p_bounce": round(win_count / total_events, 3) if total_events > 0 else 0.0,
        "k_knocks": recent_knocks,
        "total_events": total_events,
        "win_events": win_count
    }

# =========================================================
# C. 水平サポートゾーン自動検出アルゴリズム
# =========================================================
def detect_support_zones(df: pd.DataFrame, lookback_days: int = 120, order: int = 3,
                         eps_pct: float = 0.015, half_life_days: int = 60, min_bounces: int = 1) -> list:
    df_target = df.tail(lookback_days).copy().reset_index(drop=True)
    if len(df_target) < order * 2 + 1: return []

    low_prices = df_target['Low'].values
    minima_indices = argrelextrema(low_prices, np.less, order=order)[0]
    if len(minima_indices) == 0: return []

    minima_prices = low_prices[minima_indices]
    days_ago_list = (len(df_target) - 1) - minima_indices

    eps_value = df_target['Close'].iloc[-1] * eps_pct
    clustering = DBSCAN(eps=eps_value, min_samples=1).fit(minima_prices.reshape(-1, 1))

    support_zones = []
    lambda_decay = np.log(2) / half_life_days

    for label in set(clustering.labels_):
        if label == -1: continue
        cluster_idx = np.where(clustering.labels_ == label)[0]
        if len(cluster_idx) < min_bounces: continue

        zone_prices = minima_prices[cluster_idx]
        zone_days_ago = days_ago_list[cluster_idx]
        s_zone_score = np.sum(np.exp(-lambda_decay * zone_days_ago))

        support_zones.append({
            "avg_price": round(np.mean(zone_prices), 2),
            "bounces": len(cluster_idx),
            "strength_score": round(s_zone_score, 3)
        })

    return sorted(support_zones, key=lambda x: x['strength_score'], reverse=True)

# =========================================================
# D. レジスタンスライン自動検出アルゴリズム
# =========================================================
def detect_resistance_lines(df: pd.DataFrame, lookback_days: int = 120, order: int = 3,
                            eps_pct: float = 0.015, half_life_days: int = 60, trendline_tolerance_pct: float = 0.01) -> dict:
    df_target = df.tail(lookback_days).copy().reset_index(drop=True)
    if len(df_target) < order * 2 + 1: return {"horizontal_zones": [], "trendlines": []}

    high_prices = df_target['High'].values
    maxima_indices = argrelextrema(high_prices, np.greater, order=order)[0]
    if len(maxima_indices) < 2: return {"horizontal_zones": [], "trendlines": []}

    maxima_prices = high_prices[maxima_indices]
    latest_index = len(df_target) - 1
    days_ago_list = latest_index - maxima_indices

    eps_value = df_target['Close'].iloc[-1] * eps_pct
    clustering = DBSCAN(eps=eps_value, min_samples=1).fit(maxima_prices.reshape(-1, 1))

    horizontal_zones = []
    lambda_decay = np.log(2) / half_life_days
    for label in set(clustering.labels_):
        if label == -1: continue
        cluster_idx = np.where(clustering.labels_ == label)[0]
        s_zone_score = np.sum(np.exp(-lambda_decay * days_ago_list[cluster_idx]))
        horizontal_zones.append({
            "avg_price": round(np.mean(maxima_prices[cluster_idx]), 2),
            "touches": len(cluster_idx),
            "strength_score": round(s_zone_score, 3)
        })

    trendlines = []
    points = list(zip(maxima_indices, maxima_prices))
    for p1, p2, p3 in combinations(points, 3):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
        slope = (y3 - y1) / (x3 - x1) if x3 != x1 else 0
        expected_y2 = slope * (x2 - x1) + y1
        if expected_y2 > 0 and abs(y2 - expected_y2) / expected_y2 <= trendline_tolerance_pct:
          if round(slope * (latest_index + 1 - x1) + y1, 2) < sorted(horizontal_zones, key=lambda x: x['strength_score'], reverse=True)[0]['avg_price'] and df['Close'].iloc[-1] < round(slope * (latest_index + 1 - x1) + y1, 2):
            trendlines.append({
                "projected_price": round(slope * (latest_index + 1 - x1) + y1, 2),
                "slope": round(slope, 4),
            })

    return {
        "horizontal_zones": sorted(horizontal_zones, key=lambda x: x['strength_score'], reverse=True),
        "trendlines": sorted(trendlines, key=lambda x: x['projected_price'])
    }

def cluster_trendline_prices(trendlines: list, threshold_pct: float = 0.003, base_price: float = None, min_count: int = 2) -> list:
    if not trendlines: return []
    ref = base_price if base_price else np.median([t["projected_price"] for t in trendlines])
    threshold = ref * threshold_pct

    sorted_lines = sorted(trendlines, key=lambda t: t["projected_price"])
    used = [False] * len(sorted_lines)
    clusters = []

    for i, base_line in enumerate(sorted_lines):
        if used[i]: continue
        group = [base_line]
        used[i] = True
        for j in range(i + 1, len(sorted_lines)):
            if not used[j] and (sorted_lines[j]["projected_price"] - base_line["projected_price"]) <= threshold:
                group.append(sorted_lines[j])
                used[j] = True
        if len(group) >= min_count:
            clusters.append({
                "center": round(np.mean([l["projected_price"] for l in group]), 2),
                "count": len(group)
            })
    return sorted(clusters, key=lambda c: -c["count"])

# =========================================================
# スコアリング
# =========================================================
def calculate_total_score(rsi_proj: float,
                          p_bounce_raw: float,
                          total_events: int,
                          win_events: int,
                          k_knocks: int,
                          ma_price: float,
                          bb_price: float,
                          support_zones: list,
                          resistance_lines: dict) -> int:
    score = 0.0

    if rsi_proj is not None:
        score += 30.0 * np.exp(-((rsi_proj - 35) ** 2) / (2 * 8.0 ** 2))

    if total_events > 0:
        p_adj = (win_events + 1.0) / (total_events + 2.0)
    else:
        p_adj = 0.5

    if p_adj >= 0.7:
        win_multiplier = 1.0
    elif p_adj <= 0.4:
        win_multiplier = 0.0
    else:
        win_multiplier = (p_adj - 0.4) / 0.3

    if k_knocks == 0:
        knock_multiplier = 1.0
    elif k_knocks == 1:
        knock_multiplier = 0.9
    elif k_knocks == 2:
        knock_multiplier = 0.60
    else:
        knock_multiplier = 0.30

    score += 40.0 * win_multiplier * knock_multiplier

    bb_diff_pct = abs(ma_price - bb_price) / ma_price * 100
    sigma_bb = 1.0
    score += 15.0 * np.exp(- (bb_diff_pct ** 2) / (2 * sigma_bb ** 2))

    if support_zones:
        closest_zone = min(support_zones, key=lambda z: abs(z['avg_price'] - ma_price))
        dist_pct = abs(closest_zone['avg_price'] - ma_price) / ma_price * 100

        if dist_pct <= 2.5:
            normalized_strength = min(closest_zone['strength_score'] / 3.0, 1.0)
            sigma_sup = 1.8
            distance_multiplier = np.exp(- (dist_pct ** 2) / (2 * sigma_sup ** 2))
            score += 15.0 * normalized_strength * distance_multiplier

    if resistance_lines and resistance_lines.get('horizontal_zones'):
        valid_resistances = [r['avg_price'] for r in resistance_lines['horizontal_zones'] if r['avg_price'] > ma_price]
        if valid_resistances:
            closest_res = min(valid_resistances)
            upside_space_pct = (closest_res - ma_price) / ma_price * 100

            if upside_space_pct < 5.0:
                overhead_penalty = 10.0 * (5.0 - upside_space_pct) / 5.0 # ※条件を5%に広げたため計算式も5.0ベースに補正
                score -= overhead_penalty

    return min(100, max(0, int(round(score))))

# =========================================================
# 個別解析 ＆ JSON生成関数
# =========================================================
def analyze_and_export_to_json(ticker_symbol="NVDA", ma_col="MA200", ma_label="Daily 200MA"):
    print(f"[{ticker_symbol} - {ma_label}] Analyzing...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="3y")

    if df.empty:
        print(f"  -> Skipped: No data for {ticker_symbol}")
        return None

    # 指標の計算
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA50']  = df['Close'].rolling(window=50).mean()
    df['BB_Lower_2'] = calculate_bollinger_bands(df['Close'], 20, 2.0)
    df['BB_Lower_3'] = calculate_bollinger_bands(df['Close'], 20, 3.0)
    df.dropna(inplace=True)

    if ma_col not in df.columns:
        print(f"  -> Error: Target MA '{ma_col}' not calculated.")
        return None

    current_price = df['Close'].iloc[-1]
    target_ma_price = df[ma_col].iloc[-1]

    # 🔴【改善ポイント】: 評価やファイル出力を始める前に、距離（0%〜+5%）をチェックし早期リターンする
    distance_pct = (current_price - target_ma_price) / target_ma_price * 100

    if distance_pct < 0 or distance_pct > 5.0:
        print(f"  -> ⏭️ Skipped: Price is {distance_pct:.2f}% away from {ma_col} (Out of 0% to +5% range).")
        return None # 範囲外ならここで処理を打ち切るため、無駄なファイル出力や重い計算が走らない

    # ▼ 以降は「指定MAの0%〜+5%以内にいる銘柄」のみ実行される ▼

    bb_lower_2_price = df['BB_Lower_2'].iloc[-1]
    bb_lower_3_price = df['BB_Lower_3'].iloc[-1]

    prev_close = df['Close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100

    # A〜Dの解析実行（引数 ma_col を渡すことで、指定されたMAに対して正しく計算される）
    res_A = evaluate_tomorrow_bounce_rsi(df, ma_col=ma_col)
    res_B = calculate_bounce_history(df, ma_col=ma_col)
    res_C = detect_support_zones(df)
    res_D = detect_resistance_lines(df)

    trend_clusters = cluster_trendline_prices(res_D['trendlines'], base_price=current_price)

    bb2_diff_pct = abs(target_ma_price - bb_lower_2_price) / target_ma_price * 100
    bb3_diff_pct = abs(target_ma_price - bb_lower_3_price) / target_ma_price * 100

    # スコアの計算
    total_score = calculate_total_score(
        rsi_proj=res_A.get('projected_rsi', 50),
        p_bounce_raw=res_B.get('p_bounce', 0.0),
        total_events=res_B.get('total_events', 0),
        win_events=res_B.get('win_events', 0),
        k_knocks=res_B.get('k_knocks', 0),
        ma_price=target_ma_price,
        bb_price=bb_lower_2_price,
        support_zones=res_C,
        resistance_lines=res_D
    )

    star_rating = np.floor((total_score / 10.0) + 0.5) / 2.0
    top_support = res_C[0]['avg_price'] if res_C else target_ma_price

    valid_horizontal = [{"price": z["avg_price"], "touches": z["touches"], "strength_score": z["strength_score"]}
                        for z in res_D.get('horizontal_zones', []) if z["avg_price"] > current_price]

    valid_trend_clusters = [{"price": c["center"], "count": c["count"]}
                            for c in trend_clusters if c["center"] > current_price]

    top_resistance = valid_horizontal[0]['price'] if valid_horizontal else (current_price * 1.1)

    chart_df = df.tail(150)
    k_knocks = res_B.get('k_knocks', 0)
    proj_rsi = res_A.get('projected_rsi', 50)
    company_name = ticker.info.get('shortName', ticker_symbol)

    json_data = {
        "meta": {
            "ticker": ticker_symbol,
            "name": company_name,
            "market": ticker.info.get('exchange', 'US'),
            "price": round(current_price, 2),
            "change": round(change, 2),
            "changePercent": round(change_pct, 2)
        },
        "score": {
            "total": total_score,
            "stars": star_rating,
            "note": f"/ 100 (Win Rate: {res_B.get('p_bounce',0)*100:.0f}%)"
        },
        "raw_metrics": {
            "targetMA_label": ma_label,
            "targetMA_price": round(target_ma_price, 2),
            "maDistance_pct": round(distance_pct, 2),
            "bb2Confluence_pct": round(bb2_diff_pct, 2),
            "bb3Confluence_pct": round(bb3_diff_pct, 2),
            "maKnock_count": k_knocks,
            "expectedRsi": proj_rsi
        },
        "metrics": {
            "maDistance": f"{'+' if distance_pct > 0 else ''}{distance_pct:.2f}%",
            "bbConfluence": f"{bb2_diff_pct:.2f}%",
            "maKnock": f"{k_knocks} time{'s' if k_knocks != 1 else ''} recently",
            "expectedRsi": f"{proj_rsi}"
        },
        "resistances": {
            "horizontal": valid_horizontal,
            "trendline_clusters": valid_trend_clusters
        },
        "summary": f"The SwingRatings algorithm detects that {ticker_symbol} is currently testing its {ma_label} near ${round(target_ma_price, 2)}. Historically, the win rate for bounces at this support level is {res_B.get('p_bounce',0)*100:.1f}%. In the event of a successful bounce, the primary overhead resistance is projected around ${round(top_resistance, 2)}.",
        "chartData": {
            "candles": [{"time": str(idx.date()), "open": row.Open, "high": row.High, "low": row.Low, "close": row.Close} for idx, row in chart_df.iterrows()],
            "ma_target": [{"time": str(idx.date()), "value": row[ma_col]} for idx, row in chart_df.iterrows()],
            "bbLower": [{"time": str(idx.date()), "value": row.BB_Lower_2} for idx, row in chart_df.iterrows()],
            "supportLine": top_support
        }
    }

    filename = f"{ticker_symbol.lower()}_{ma_col.lower()}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    summary_item = {
        "ticker": ticker_symbol,
        "name": company_name,
        "price": round(current_price, 2),
        "changePercent": round(change_pct, 2),
        "score": total_score,
        "stars": star_rating,
        "ma_label": ma_label,
        "ma_col": ma_col.lower(),
        "ma_distance": round(distance_pct, 2),
        "bb_confluence": round(bb2_diff_pct, 2),
        "expected_rsi": proj_rsi
    }
    return summary_item

# =========================================================
# 実行エントリーポイント
# =========================================================
if __name__ == "__main__":

    tickers = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'BRK.B', 'TSLA', 'AVGO',
        'LLY', 'WMT', 'JPM', 'XOM', 'V', 'UNH', 'MA', 'ORCL', 'PG', 'HD',
        'COST', 'JNJ', 'NFLX', 'ABBV', 'CVX', 'BAC', 'KO', 'MRK', 'PEP', 'ADBE',
        'CRM', 'PM', 'TMO', 'LIN', 'WFC', 'CSCO', 'ACN', 'AMD', 'MCD', 'ABT',
        'GE', 'DIS', 'ISRG', 'INTU', 'DHR', 'CAT', 'IBM', 'TXN', 'AXP', 'VZ',
        'AMAT', 'MS', 'PFE', 'CMCSA', 'NEE', 'HON', 'QCOM', 'LOW', 'AMGN', 'UNP',
        'SPGI', 'RTX', 'GS', 'SYK', 'NOW', 'COP', 'INTC', 'LRCX', 'BLK', 'ELV',
        'TJX', 'ETN', 'UPS', 'DE', 'C', 'LMT', 'BSX', 'BA', 'MDT', 'PGR',
        'ADP', 'CB', 'VRTX', 'ADI', 'CI', 'BMY', 'PLD', 'MU', 'REGN', 'MMC',
        'SNE', 'MDLZ', 'CVS', 'GILD', 'AMT', 'SCHW', 'ZTS', 'ICE', 'APH', 'WM',
        'BX', 'T', 'GEV', 'PANW', 'SNPS', 'KLAC', 'CDNS', 'SHW', 'MO', 'EOG',
        'MCK', 'MSTR', 'DUK', 'ORLY', 'ITW', 'CL', 'NXPI', 'SLB', 'MAR', 'ANET',
        'GD', 'ECL', 'BDX', 'EMR', 'ABNB', 'CTAS', 'AON', 'HUM', 'MCO', 'APH',
        'USB', 'ROP', 'PH', 'NOC', 'ADSK', 'A', 'PNC', 'BSX', 'TGT', 'HCA',
        'TDG', 'EW', 'MSI', 'FDX', 'D', 'WELL', 'CHTR', 'PCAR', 'CSX', 'TT',
        'PSX', 'AJG', 'MET', 'CPRT', 'NEM', 'O', 'OKE', 'MAR', 'KMB', 'TRV',
        'VLO', 'ECL', 'NSC', 'BK', 'MCHP', 'SRE', 'PSA', 'COR', 'AIG', 'SO',
        'CARR', 'F', 'STZ', 'PAYX', 'DLR', 'TEL', 'O', 'AEP', 'ADM', 'AZO',
        'RMD', 'IQV', 'PRU', 'NKE', 'JCI', 'CTVA', 'FIS', 'KVUE', 'MNST', 'CNC',
        'GWW', 'DOW', 'EXC', 'EL', 'BKR', 'KDP', 'ED', 'BBY', 'CDW', 'KEYS',
        'AFL', 'CMI', 'ALN', 'PCG', 'HES', 'MPC', 'FMC', 'GLW', 'KMI', 'WMB',
        'DHI', 'HLT', 'HPE', 'HPQ', 'K', 'KHC', 'KR', 'LEN', 'LH', 'MGM',
        'NDAQ', 'NWL', 'OXY', 'PAYC', 'PYPL', 'RCL', 'ROK', 'SBUX', 'STT', 'SYY',
        'TRGP', 'UDR', 'VMC', 'WBA', 'WDC', 'WHR', 'WY', 'WYNN', 'XEL', 'ZBH',
        'ZION', 'AWK', 'BXP', 'DFS', 'DISH', 'DRI', 'FITB', 'GPN', 'HAS', 'HRL',
        'IVZ', 'KEY', 'LVS', 'MTB', 'NTRS', 'RF', 'TROW', 'VTR', 'BEN', 'CPT',
        'EQR', 'ESS', 'FRT', 'HST', 'IRM', 'MAA', 'UDR', 'AVB', 'ARE', 'BBY',
        'COO', 'EFX', 'EPAM', 'EXPD', 'FDS', 'FFIV', 'GRMN', 'IDXX', 'JKHY', 'OTIS',
        'POOL', 'ROL', 'TER', 'TYL', 'VRSN', 'WAT', 'ZBRA', 'AES', 'ATO', 'CNP',
        'FE', 'LNT', 'NI', 'NRG', 'PEG', 'PNW', 'PPL', 'VST', 'WEC', 'MTCH',
        'SWKS', 'TCOM', 'WIX', 'TTD', 'OKTA', 'DDOG', 'ZS', 'MDB', 'TEAM', 'WDAY',
        'DOCU', 'ROKU', 'ZM', 'PTON', 'SHOP', 'SQ', 'COIN', 'PLTR', 'U', 'SNOW',
        'ABG', 'ACM', 'AECOM', 'AEE', 'AEG', 'AEM', 'AER', 'AFG', 'AGCO', 'AGI',
        'AGL', 'AGO', 'AGR', 'AIB', 'AIG', 'AIZ', 'AJG', 'AKAM', 'AL', 'ALB',
        'ALGN', 'ALK', 'ALL', 'ALLE', 'ALLY', 'ALNY', 'ALV', 'AMCR', 'AME', 'AMH',
        'AMK', 'AMP', 'AMRC', 'AMWD', 'AN', 'ANF', 'ANTM', 'AOS', 'APA', 'APD',
        'APO', 'APTV', 'ARE', 'ARNC', 'ARW', 'ASB', 'ASML', 'ASPN', 'ASR', 'ASX',
        'ATH', 'ATO', 'ATR', 'ATVI', 'AVA', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP',
        'AYI', 'AZO', 'BA', 'BABA', 'BAC', 'BAH', 'BALL', 'BAX', 'BBWI', 'BBY',
        'BC', 'BCE', 'BCO', 'BDX', 'BEN', 'BERY', 'BF.B', 'BHF', 'BIIB', 'BIO',
        'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BLL', 'BMO', 'BNS', 'BP', 'BR',
        'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB',
        'CBOE', 'CBRE', 'CCI', 'CCK', 'CCL', 'CDNS', 'CDW', 'CE', 'CELG', 'CERN',
        'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
        'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'CNQ', 'COF', 'COO',
        'COP', 'COST', 'CPB', 'CPRI', 'CPRT', 'CRM', 'CRWD', 'CSCO', 'CSX', 'CTAS',
        'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD',
        'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISH', 'DLR', 'DLTR',
        'DOV', 'DOW', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC',
        'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR',
        'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY',
        'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FBHS',
        'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC',
        'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS',
        'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS',
        'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT',
        'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM',
        'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU',
        'IP', 'IPG', 'IQV', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT',
        'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY', 'KEYS', 'KHC', 'KIM',
        'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L', 'LDOS', 'LEN', 'LH',
        'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN',
        'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD',
        'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC',
        'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK',
        'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
        'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NLOK', 'NLSN', 'NOC',
        'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS',
        'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OI', 'OKE', 'OMC', 'ORCL', 'ORLY',
        'OTIS', 'OXY', 'PAYC', 'PAYX', 'PBCT', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PENN'
    ]
    columns = ['MA200', 'MA100', 'MA50']

    label_map = {
        'MA200': 'Daily 200MA',
        'MA100': 'Daily 100MA',
        'MA50':  'Daily 50MA'
    }

    screener_data = []

    print("🚀 SwingRatings Batch Processing Started...")

    for ticker in tickers:
        for col in columns:
            try:
                ma_label = label_map.get(col, col)

                # ここで列名（MA100やMA50など）を引数として渡すため、関数内でも対象のMAが使われます
                summary = analyze_and_export_to_json(
                    ticker_symbol=ticker,
                    ma_col=col,
                    ma_label=ma_label
                )

                if summary:
                    screener_data.append(summary)

            except Exception as e:
                print(f"  ❌ Error processing {ticker} ({col}): {e}")

    screener_data = sorted(screener_data, key=lambda x: x["score"], reverse=True)

    # 2. 🟢 重複削除ロジックを追加 (ticker と ma_col の組み合わせで一意にする)
    unique_screener_data = []
    seen_pairs = set()

    for entry in screener_data:
      # ticker(銘柄名)とma_col(移動平均の種類)のペアを作成
      pair = (entry['ticker'], entry['ma_col'])
      
      # まだ記録されていないペアであれば追加
      if pair not in seen_pairs:
          unique_screener_data.append(entry)
          seen_pairs.add(pair)

    # 重複を除去したデータで上書き
    screener_data = unique_screener_data

    # 3. screener.json として一括保存
    with open("screener.json", "w", encoding="utf-8") as f:
      json.dump(screener_data, f, ensure_ascii=False, indent=2)

    print("\n✅ All processing complete!")
    print(f"🎉 Generated screener.json with {len(screener_data)} qualified signals.")