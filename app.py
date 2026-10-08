import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
import requests
import re
import time
import hashlib
from io import StringIO
from io import BytesIO
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# 頁面基本設定
st.set_page_config(
    page_title="HKJC AI 智能賽馬預測系統",
    page_icon="🐎",
    layout="wide"
)

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統 (賽道移欄解析 + 凱利注碼)")
st.markdown("---")

# 初始化模擬投注紀錄與單場凍結名單
if 'simulated_bets' not in st.session_state:
    st.session_state['simulated_bets'] = pd.DataFrame(columns=[
        '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
    ])
if 'frozen_races' not in st.session_state:
    st.session_state['frozen_races'] = {}  # 格式: {race_no: "HH:MM:SS"}

# 設定 repo 內的模型及預測 CSV 路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'
PREDICTION_CSV_PATH = APP_DIR / 'prediction.csv'
FEATURE_COLS_15 = [
    'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
    '排位檔位', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    'horse_win_rate', 'horse_last_rank',
    '距離', 'horse_surface_win_rate', 'horse_dist_win_rate',
]

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"⚠️ 找不到 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

@st.cache_data
def load_historical_stats(history_csv_path='hkjc_all_seasons_features.csv'):
    """從歷史大表中計算騎師、練馬師、馬匹勝率，以及馬匹上仗名次、場地勝率、路程勝率"""
    csv_path = Path(__file__).resolve().parent / history_csv_path
    if not os.path.exists(csv_path):
        return None
    
    try:
        hist_df = pd.read_csv(csv_path, low_memory=False)
        hist_df['numeric_rank'] = pd.to_numeric(hist_df['名次'], errors='coerce')
        hist_df['is_win'] = (hist_df['numeric_rank'] == 1).astype(int)
        
        hist_df['_clean_jockey'] = hist_df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        hist_df['_clean_trainer'] = hist_df['練馬師'].astype(str).str.replace(' ', '')
        hist_df['_clean_horse'] = hist_df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        
        jockey_stats = hist_df.groupby('_clean_jockey', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_jockey_win_rate'})
        trainer_stats = hist_df.groupby('_clean_trainer', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_trainer_win_rate'})
        horse_stats = hist_df.groupby('_clean_horse', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_horse_win_rate'})
        combo_stats = hist_df.groupby(['_clean_jockey', '_clean_trainer'], as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_combo_win_rate'})
        
        valid_ranks_df = hist_df.dropna(subset=['numeric_rank']).copy()
        if '賽事編號' in valid_ranks_df.columns:
            valid_ranks_df = valid_ranks_df.sort_values('賽事編號')
        horse_last_rank_stats = (
            valid_ranks_df.groupby('_clean_horse', as_index=False)
            .last()[['_clean_horse', 'numeric_rank']]
            .rename(columns={'numeric_rank': 'hist_horse_last_rank'})
        )
        
        if '場地' in hist_df.columns:
            hist_df['_clean_surface'] = hist_df['場地'].astype(str).str.strip()
            horse_surface_stats = (
                hist_df.groupby(['_clean_horse', '_clean_surface'], as_index=False)['is_win']
                .mean()
                .rename(columns={'is_win': 'hist_horse_surface_win_rate'})
            )
        else:
            horse_surface_stats = pd.DataFrame(columns=['_clean_horse', '_clean_surface', 'hist_horse_surface_win_rate'])
            
        if '距離' in hist_df.columns:
            hist_df['_clean_dist'] = pd.to_numeric(hist_df['距離'], errors='coerce').fillna(1200).astype(int)
            horse_dist_stats = (
                hist_df.groupby(['_clean_horse', '_clean_dist'], as_index=False)['is_win']
                .mean()
                .rename(columns={'is_win': 'hist_horse_dist_win_rate'})
            )
        else:
            horse_dist_stats = pd.DataFrame(columns=['_clean_horse', '_clean_dist', 'hist_horse_dist_win_rate'])
            
        return (
            jockey_stats, trainer_stats, horse_stats, combo_stats,
            horse_last_rank_stats, horse_surface_stats, horse_dist_stats
        )
    except Exception as e:
        st.error(f"載入歷史數據失敗：{e}")
        return None

def find_racecard_table(html_text):
    try:
        tables = pd.read_html(StringIO(html_text))
    except ImportError as exc:
        raise RuntimeError('缺少 pandas HTML 解析依賴。') from exc
    except ValueError:
        return None
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = ['_'.join(map(str, col)) for col in table.columns]
        table.columns = [str(col).replace('\n', ' ').strip() for col in table.columns]
        header_text = ' '.join(table.columns)
        if '馬名' in header_text and '騎師' in header_text:
            return table
    return None

def find_column(columns, keywords, fallback=None):
    return next((col for col in columns if any(word in str(col) for word in keywords)), fallback)

def fetch_hkjc_racecard(race_date):
    date_text = race_date.strftime('%Y/%m/%d')
    race_day_id = race_date.strftime('%Y%m%d')
    season_start = race_date.year if race_date.month >= 9 else race_date.year - 1
    season = f"{season_start % 100:02d}/{(season_start + 1) % 100:02d}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    session = requests.Session()
    rows = []
    race_venue = None
    first_race = None
    
    for venue_code in ('ST', 'HV'):
        url = f'https://racing.hkjc.com/racing/information/Chinese/Racing/RaceCard.aspx?racedate={date_text}&Racecourse={venue_code}&RaceNo=1'
        response = session.get(url, headers=headers, timeout=12)
        response.encoding = 'utf-8'
        first_race_table = find_racecard_table(response.text)
        if first_race_table is not None:
            race_venue = venue_code
            first_race = (first_race_table, response.text)
            break
            
    if first_race is None:
        raise ValueError('HKJC 暫未提供該日期的賽卡。')

    for race_no in range(1, 16):
        if race_no == 1:
            race_table, race_html = first_race
        else:
            url = f'https://racing.hkjc.com/racing/information/Chinese/Racing/RaceCard.aspx?racedate={date_text}&Racecourse={race_venue}&RaceNo={race_no}'
            response = session.get(url, headers=headers, timeout=12)
            response.encoding = 'utf-8'
            race_html = response.text
            race_table = find_racecard_table(race_html)
            if race_table is None:
                break

        distance_match = re.search(r'(\d{4})\s*米', race_html)
        distance = int(distance_match.group(1)) if distance_match else 1200
        surface = '泥地' if ('泥地' in race_html or '全天候' in race_html) else '草地'
        
        # 🚀 解析賽道移欄 (A/B/C/C+3 賽道)
        track_match = re.search(r'"([A-Z0-9\+]+)"\s*賽道', race_html)
        course_track = track_match.group(1) if track_match else ('泥地' if surface == '泥地' else 'A')

        columns = list(race_table.columns)
        horse_no_col = find_column(columns, ('馬號', '編號'), columns[0] if columns else None)
        horse_name_col = find_column(columns, ('馬名',))
        weight_col = find_column(columns, ('負磅', '磅'))
        jockey_col = find_column(columns, ('騎師',))
        trainer_col = find_column(columns, ('練馬師',))
        draw_col = find_column(columns, ('檔位',))
        if not all((horse_no_col, horse_name_col, weight_col, jockey_col, trainer_col, draw_col)):
            continue

        for _, row in race_table.iterrows():
            horse_no = str(row[horse_no_col]).strip()
            if re.fullmatch(r'\d+\.0', horse_no): horse_no = horse_no[:-2]
            if not horse_no.isdigit(): continue
            horse_name = re.sub(r'\(.*?\)', '', str(row[horse_name_col])).strip()
            if not horse_name or horse_name.lower() == 'nan': continue
            
            rows.append({
                '馬季': season,
                '賽事編號': f'{race_day_id}-{race_no:02d}',
                '名次': '',
                '馬號': horse_no,
                '馬名': horse_name,
                '騎師': str(row[jockey_col]).strip(),
                '練馬師': str(row[trainer_col]).strip(),
                '實際負磅': str(row[weight_col]).strip(),
                '排位檔位': str(row[draw_col]).strip(),
                '獨贏賠率': 10.0,
                '位置賠率': 1.5,
                '距離': distance,
                '場地': surface,
                '賽道': course_track,
                'racecourse_code': race_venue,
            })
        time.sleep(0.25)

    if not rows: raise ValueError('HKJC 沒有回傳排位表。')
    return pd.DataFrame(rows).drop_duplicates(['賽事編號', '馬號'], keep='last').reset_index(drop=True)

# ==========================================
# 📂 預測資料 (支援多檔案即時切換與抓取)
# ==========================================
st.sidebar.header("📂 預測資料")
available_csvs = sorted([f.name for f in APP_DIR.glob("prediction*.csv")])

if not available_csvs:
    selected_csv_name = "prediction.csv"
else:
    selected_csv_name = st.sidebar.selectbox("📜 選擇 Repo 內的預測賽卡", available_csvs, key="repo_csv_selector")

if st.session_state.get('active_csv_name') != selected_csv_name:
    st.session_state['active_csv_name'] = selected_csv_name
    st.session_state.pop('fetched_prediction_df', None)
    st.session_state.pop('df_data', None)
    st.session_state.pop('prediction_signature', None)
    st.session_state.pop('baseline_odds_map', None)
    st.session_state.pop('baseline_locked_time', None)
    st.session_state['frozen_races'] = {}
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1

SELECTED_CSV_PATH = APP_DIR / selected_csv_name

race_date_choice = st.sidebar.date_input('選擇賽事日期 (抓新排位用)', value=datetime.now().date(), key='racecard_date_choice')
if st.sidebar.button('🏇 抓取排位並載入預測', use_container_width=True, key='fetch_racecard_button'):
    try:
        with st.spinner(f'正在抓取 HKJC {race_date_choice:%Y-%m-%d} 排位表…'):
            fetched_card = fetch_hkjc_racecard(race_date_choice)
        st.session_state['fetched_prediction_df'] = fetched_card
        st.session_state['fetched_prediction_signature'] = f"hkjc:{race_date_choice:%Y%m%d}:{len(fetched_card)}"
        st.session_state['fetched_prediction_date'] = race_date_choice
        st.session_state['df_data'] = fetched_card.copy()
        st.session_state['prediction_signature'] = st.session_state['fetched_prediction_signature']
        st.session_state['pending_auto_odds_fetch'] = True
        st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
        st.session_state.pop('odds_last_status', None)
        st.session_state.pop('odds_last_error', None)
        st.session_state.pop('baseline_odds_map', None)
        st.session_state.pop('baseline_locked_time', None)
        st.session_state['frozen_races'] = {}
        st.sidebar.success(f'已載入 {len(fetched_card)} 匹馬。')
    except Exception as exc:
        st.sidebar.error(f'抓取排位失敗：{exc}')

current_fetched_card = st.session_state.get('fetched_prediction_df')
if current_fetched_card is not None:
    if st.sidebar.button('🔄 放棄抓取資料，改用上方選單 CSV', use_container_width=True):
        st.session_state.pop('fetched_prediction_df', None)
        st.session_state.pop('df_data', None)
        st.session_state.pop('prediction_signature', None)
        st.session_state.pop('baseline_odds_map', None)
        st.session_state.pop('baseline_locked_time', None)
        st.session_state['frozen_races'] = {}
        st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
        st.rerun()

backtest_file = st.sidebar.file_uploader(
    "回測用：上傳已完成賽事 CSV",
    type=['csv'],
    key='backtest_results_upload'
)

# ==========================================
# ⚙️ 策略、資金與算分引擎設定
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("⚙ 投注策略與算分引擎")
strategy_mode = st.sidebar.radio(
    "1️⃣ 推薦排序核心邏輯",
    ["⚡ 傳統純 EV 排序 (高賠率優先)", "🛡️ 抗噪穩定 (基本面優先 + 排除回飛)", "🎯 純 AI 勝率最高 (命中率優先)"],
    index=0
)
use_dynamic_form = st.sidebar.checkbox("2️⃣ 將「上仗名次與同程勝率」納入算分", value=False)
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 6.0, 2.5, 0.1)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=25.0)

st.sidebar.markdown("---")
st.sidebar.header("💰 資金管理 (凱利注碼精算)")
total_bankroll = st.sidebar.number_input("💵 單日總本金預算 (HKD)", min_value=100, max_value=1000000, value=5000, step=500)
kelly_fraction = st.sidebar.slider("🎛️ 凱利保守係數", 0.05, 1.0, 0.15, 0.05)

# ==========================================
# 🛠️ 歷史賽果抓取 (加入賽道解析)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🛠️ 歷史賽果 15 大特徵抓取器")
tool_date_choice = st.sidebar.date_input("選擇目標賽事日期", value=datetime.now().date(), key="tool_date_input")
tool_season = st.sidebar.text_input("輸入馬季", value="26/27")

if st.sidebar.button("📥 抓取並產出完賽 CSV", use_container_width=True, key="run_tool_button"):
    date_str_tool = tool_date_choice.strftime('%Y/%m/%d')
    date_for_id_tool = tool_date_choice.strftime('%Y%m%d')
    all_races_data_tool = []
    headers_tool = {'User-Agent': 'Mozilla/5.0'}
    progress_bar = st.sidebar.progress(0)
    
    for race_no in range(1, 13):
        url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={date_str_tool}&RaceNo={race_no}"
        try:
            resp = requests.get(url, headers=headers_tool, timeout=15)
            resp.encoding = 'utf-8' 
            html_text = resp.text
            if "沒有相關資料" in html_text:
                progress_bar.progress(race_no / 12)
                continue

            dist_match = re.search(r'(\d{4})\s*米', html_text)
            distance = int(dist_match.group(1)) if dist_match else 1200
            surface_type = "草地" if "草地" in html_text else "泥地"
            
            # 🚀 抓取賽果網頁的賽道移欄資訊
            track_match = re.search(r'"([A-Z0-9\+]+)"\s*賽道', html_text)
            course_track = track_match.group(1) if track_match else ('泥地' if surface_type == '泥地' else 'A')
            
            tables = pd.read_html(StringIO(html_text))
            target_df = None
            for t in tables:
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = ['_'.join(map(str, col)) for col in t.columns]
                check_str = "".join([str(c) for c in t.columns])
                for row_idx in range(min(5, len(t))):
                    check_str += "".join([str(c) for c in t.iloc[row_idx].values])
                if '名次' in check_str and '馬號' in check_str and '獨贏' in check_str:
                    if not any('名次' in str(c) for c in t.columns):
                        for row_idx in range(min(5, len(t))):
                            row_str = "".join([str(c) for c in t.iloc[row_idx].values])
                            if '名次' in row_str and '馬號' in row_str:
                                t.columns = [str(c) for c in t.iloc[row_idx]]
                                t = t.drop(range(row_idx + 1)).reset_index(drop=True) 
                                break
                    target_df = t
                    break
                    
            if target_df is not None and not target_df.empty:
                cols = target_df.columns.astype(str)
                rank_col = next((col for col in cols if '名次' in col), None)
                horse_no_col = next((col for col in cols if '馬號' in col or '編號' in col), None)
                odds_col = next((col for col in cols if '獨贏' in col), None)
                draw_col = next((col for col in cols if '檔位' in col), None)
                
                if rank_col and horse_no_col:
                    for idx, row in target_df.iterrows():
                        horse_no = str(row.get(horse_no_col, '')).strip()
                        if horse_no.endswith('.0'): horse_no = horse_no[:-2]
                        if horse_no.isdigit():
                            all_races_data_tool.append({
                                '賽事編號': f"{date_for_id_tool}-{race_no:02d}",
                                '名次': str(row.get(rank_col, '')).strip(),
                                '馬號': horse_no,
                                '馬名': str(row.get(next((c for c in cols if '馬名' in c), ''), '')).split('(')[0].strip(),
                                '騎師': str(row.get(next((c for c in cols if '騎師' in c), ''), '')).strip(),
                                '練馬師': str(row.get(next((c for c in cols if '練馬師' in c), ''), '')).strip(),
                                '實際負磅': str(row.get(next((c for c in cols if '負磅' in c or '實際負磅' in c), ''), '')).strip(),
                                '排位檔位': str(row[draw_col]).strip() if draw_col and pd.notna(row[draw_col]) else '7',
                                '獨贏賠率': row[odds_col] if odds_col and pd.notna(row[odds_col]) else 0.0,
                                '距離': distance,                
                                '場地': surface_type,
                                '賽道': course_track,
                            })
            progress_bar.progress(race_no / 12)
            time.sleep(1.0)
        except Exception: continue
    progress_bar.empty()
    
    if all_races_data_tool:
        df_tool = pd.DataFrame(all_races_data_tool)
        csv_data = df_tool.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.sidebar.success("🎉 抓取成功！")
        st.sidebar.download_button("⬇️ 下載完賽 CSV", data=csv_data, file_name=f"hkjc_data_{date_for_id_tool}.csv", mime="text/csv", use_container_width=True)

# ==========================================
# HKJC 新版 GraphQL 賠率抓取
# ==========================================
GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"
ODDS_QUERY = """query racing($date: String,$venueCode: String, $oddsTypes: [OddsType],$raceNo: Int) { raceMeetings(date: $date, venueCode:$venueCode) { pmPools(oddsTypes: $oddsTypes, raceNo:$raceNo) { oddsType lastUpdateTime leg { races } oddsNodes { combString oddsValue } } } }"""
GRAPHQL_HEADERS = { "Accept": "*/*", "Content-Type": "application/json", "Origin": "https://bet.hkjc.com", "Referer": "https://bet.hkjc.com/", "User-Agent": "Mozilla/5.0"}

def normalize_horse_no(value):
    try: return str(int(float(str(value).strip())))
    except: return str(value).strip()

def extract_race_no(value):
    match = re.search(r"-(\d+)\s*$", str(value).strip())
    return int(match.group(1)) if match else None

def normalize_race_id(value):
    match = re.search(r"(20\d{6}).*?(\d+)\s*$", str(value).strip())
    return f"{match.group(1)}-{int(match.group(2)):02d}" if match else str(value).strip()

def fetch_live_odds(date_str, venue):
    try:
        resp = requests.post(GRAPHQL_URL, headers=GRAPHQL_HEADERS, json={"operationName": "racing", "variables": {"date": date_str, "venueCode": venue, "raceNo": None, "oddsTypes": ["WIN", "PLA"]}, "query": ODDS_QUERY}, timeout=20)
        resp.raise_for_status()
        meetings = (resp.json().get("data") or {}).get("raceMeetings") or []
        if not meetings: return None, None, "無資料"
        odds_by_race, timestamps = {}, []
        for pool in meetings[0].get("pmPools") or []:
            pool_type = str(pool.get("oddsType", "")).upper()
            if pool_type not in ("WIN", "PLA"): continue
            if pool.get("lastUpdateTime"): timestamps.append(pool["lastUpdateTime"])
            for race_number in (pool.get("leg") or {}).get("races") or []:
                market = odds_by_race.setdefault(int(race_number), {"WIN": {}, "PLA": {}})
                for node in pool.get("oddsNodes") or []:
                    horse_no = normalize_horse_no(node.get("combString"))
                    if horse_no and str(node.get("oddsValue")).strip() not in ("None", "", "SCR"):
                        try: market[pool_type][horse_no] = float(str(node.get("oddsValue")).strip())
                        except: pass
        return odds_by_race, (max(timestamps) if timestamps else None), None
    except Exception as exc: return None, None, f"錯誤：{exc}"

# ==========================================
# 決定讀取賽卡資料
# ==========================================
fetched_prediction_df = st.session_state.get('fetched_prediction_df')
hk_timezone = timezone(timedelta(hours=8))

if fetched_prediction_df is not None:
    df_raw = fetched_prediction_df.copy()
    prediction_signature = st.session_state.get('fetched_prediction_signature', "HKJC_fetch")
elif SELECTED_CSV_PATH.is_file():
    try: df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='utf-8-sig')
    except: df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='cp950')
    prediction_signature = f"{SELECTED_CSV_PATH.name}:{SELECTED_CSV_PATH.stat().st_mtime_ns}"
    st.sidebar.success(f"✅ 載入：`{SELECTED_CSV_PATH.name}` ({len(df_raw)}匹)")
else: st.stop()

if fetched_prediction_df is not None and st.session_state.pop('pending_auto_odds_fetch', False):
    odds_date = st.session_state.get('fetched_prediction_date', race_date_choice).strftime('%Y-%m-%d')
    odds_by_race, _, _ = fetch_live_odds(odds_date, str(df_raw['racecourse_code'].iloc[0]))
    if odds_by_race:
        for idx, runner in df_raw.iterrows():
            market = odds_by_race.get(extract_race_no(runner.get('賽事編號', '')), {})
            win_price = market.get('WIN', {}).get(normalize_horse_no(runner.get('馬號', '')))
            if win_price: df_raw.at[idx, '獨贏賠率'] = win_price
            pla_price = market.get('PLA', {}).get(normalize_horse_no(runner.get('馬號', '')))
            if pla_price: df_raw.at[idx, '位置賠率'] = pla_price
        st.session_state['df_data'] = df_raw.copy()
        st.session_state['baseline_odds_map'] = {f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}": float(r.get('獨贏賠率',10)) for _,r in df_raw.iterrows()}
        st.session_state['baseline_locked_time'] = datetime.now(hk_timezone).strftime("%H:%M:%S")

if '獨贏賠率' not in df_raw.columns: df_raw['獨贏賠率'] = 10.0
if '位置賠率' not in df_raw.columns: df_raw['位置賠率'] = 1.0 + (pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0) - 1.0) / 3.2
if '賽道' not in df_raw.columns: df_raw['賽道'] = 'A'

if 'df_data' not in st.session_state or st.session_state.get('prediction_signature') != prediction_signature:
    st.session_state['df_data'] = df_raw.copy()
    st.session_state['prediction_signature'] = prediction_signature
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
    st.session_state['baseline_odds_map'] = {f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}": float(pd.to_numeric(r['獨贏賠率'], errors='coerce')) for _,r in df_raw.iterrows()}
    st.session_state['baseline_locked_time'] = "初始載入值"

# ==========================================
# 互動面板與特徵計算
# ==========================================
df_editable = st.session_state['df_data'][['賽事編號', '馬號', '馬名', '排位檔位', '賽道', '獨贏賠率', '位置賠率']].copy()
edited_df = st.data_editor(df_editable, disabled=['賽事編號', '馬號', '馬名', '排位檔位', '賽道'], use_container_width=True, hide_index=True, key=f"odds_editor_{st.session_state['odds_editor_version']}")
st.session_state['df_data']['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
st.session_state['df_data']['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce').fillna(1.5)

df = st.session_state['df_data'].copy()

# 落飛計算
df['_horse_uid'] = df.apply(lambda r: f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}", axis=1)
df['基準賠率'] = pd.to_numeric(df['_horse_uid'].map(st.session_state.get('baseline_odds_map', {})).fillna(df['獨贏賠率']), errors='coerce')
df['odds_drop_pct'] = ((df['基準賠率'] - df['獨贏賠率']) / df['基準賠率'].replace(0, np.nan)).fillna(0.0) * 100.0
df['資金流向'] = df['odds_drop_pct'].apply(lambda pct: f"🔥 大落飛 (-{pct:.0f}%)" if pct >= 20 else (f"📉 落飛 (-{pct:.0f}%)" if pct >= 10 else (f"🥶 回飛 (+{abs(pct):.0f}%)" if pct <= -20 else "➖ 穩定")))

# 15特徵
df['market_prob'] = 1 / df['獨贏賠率']
df['market_implied_prob'] = df['market_prob'] / df.groupby('賽事編號')['market_prob'].transform('sum')
df['odds_rank'] = df.groupby('賽事編號')['獨贏賠率'].rank(method='min')
df['is_favorite'] = (df['odds_rank'] == 1).astype(int)
df['排位檔位'] = pd.to_numeric(df['排位檔位'], errors='coerce').fillna(7)
df['實際負磅'] = pd.to_numeric(df.get('實際負磅', 120), errors='coerce').fillna(120)
df['weight_diff'] = df['實際負磅'] - df.groupby('賽事編號')['實際負磅'].transform('mean')
df['weight_rank'] = df.groupby('賽事編號')['實際負磅'].rank(ascending=False, method='min')
df['距離'] = pd.to_numeric(df.get('距離', 1200), errors='coerce').fillna(1200).astype(int)
df['場地'] = df.get('場地', '草地')

hist_stats = load_historical_stats()
if hist_stats is not None:
    jockey_stats, trainer_stats, horse_stats, combo_stats, hr_stats, sur_stats, dist_stats = hist_stats
    df['_clean_jockey'] = df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_trainer'] = df['練馬師'].astype(str).str.replace(' ', '')
    df['_clean_horse'] = df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_surface'] = df['場地'].astype(str).str.strip()
    df['_clean_dist'] = df['距離'].astype(int)
    
    df = df.merge(jockey_stats, on='_clean_jockey', how='left').merge(trainer_stats, on='_clean_trainer', how='left')
    df = df.merge(horse_stats, on='_clean_horse', how='left').merge(combo_stats, on=['_clean_jockey', '_clean_trainer'], how='left')
    df = df.merge(hr_stats, on='_clean_horse', how='left').merge(sur_stats, on=['_clean_horse', '_clean_surface'], how='left').merge(dist_stats, on=['_clean_horse', '_clean_dist'], how='left')
    
    df['jockey_win_rate'] = df['hist_jockey_win_rate'].fillna(0.08)
    df['trainer_win_rate'] = df['hist_trainer_win_rate'].fillna(0.08)
    df['horse_win_rate'] = df['hist_horse_win_rate'].fillna(0.05)
    df['combo_win_rate'] = df['hist_combo_win_rate'].fillna(0.05)
    
    if use_dynamic_form:
        df['horse_last_rank'] = df['hist_horse_last_rank'].fillna(6.5)
        df['horse_surface_win_rate'] = df['hist_horse_surface_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
        df['horse_dist_win_rate'] = df['hist_horse_dist_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    else:
        df['horse_last_rank'] = 6.0
        df['horse_surface_win_rate'] = 0.08
        df['horse_dist_win_rate'] = 0.08
else:
    for c, v in [('jockey_win_rate',0.08),('trainer_win_rate',0.08),('combo_win_rate',0.05),('horse_win_rate',0.05),('horse_last_rank',6.0),('horse_surface_win_rate',0.08),('horse_dist_win_rate',0.08)]:
        df[c] = df.get(c, v)

X_predict = df[FEATURE_COLS_15].fillna(0)
df['pred_win_prob'] = model.predict_proba(X_predict)[:, 1]

# 🚀 新增：跑道移欄與檔位偏差懲罰 (Track Course Bias)
def calculate_track_bias(row):
    penalty = 0.0
    if row['場地'] == '草地':
        draw = int(row['排位檔位'])
        track = str(row.get('賽道', 'A')).upper()
        dist = int(row['距離'])
        rc = str(row.get('racecourse_code', 'ST'))
        
        # C 或 C+3 窄賽道：大外檔極度劣勢
        if track in ['C', 'C+3', 'B+2', 'C+2']:
            if draw >= 11: penalty = -0.15
            elif draw >= 9: penalty = -0.05
            
        # 沙田 1000米直路賽：外檔(看台邊)反而有優勢
        if rc == 'ST' and dist == 1000:
            if draw >= 10: penalty = +0.10
            elif draw <= 4: penalty = -0.10
            
    return penalty

df['track_bias_adj'] = df.apply(calculate_track_bias, axis=1)

# AI勝率與EV結合落飛與檔位偏差
df['smart_score'] = df['pred_win_prob'] * (1.0 + np.clip(df['odds_drop_pct']/100.0, -0.35, 0.30) + df['track_bias_adj'])
df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

# 凱利注碼
safe_odds = np.maximum(df['獨贏賠率'], 1.01)
df['kelly_pct'] = np.clip((df['ev'] - 1.0) / (safe_odds - 1.0), 0.0, 1.0)
df['suggested_stake'] = np.floor((total_bankroll * df['kelly_pct'] * kelly_fraction) / 10) * 10

def filter_and_sort_group(group, mode, min_ev_val, min_odds_val, max_odds_val):
    cond = (group['ev'] >= min_ev_val) & (group['獨贏賠率'] >= min_odds_val) & (group['獨贏賠率'] <= max_odds_val)
    if mode.startswith("🛡️"):
        cond = cond & (group['odds_drop_pct'] > -25.0)
        sort_col = 'smart_score'
    elif mode.startswith("🎯"): sort_col = 'pred_win_prob'
    else: sort_col = 'ev'
    return group[cond].sort_values(by=sort_col, ascending=False).reset_index(drop=True)

df_backtest = df.copy()
# 👇 補上這行，確保沒有上傳檔案時有預設值：
backtest_ready = False
if backtest_file is not None:
    try:
        try: result_raw = pd.read_csv(backtest_file, encoding='utf-8-sig')
        except:
            backtest_file.seek(0)
            result_raw = pd.read_csv(backtest_file, encoding='cp950')
        result_map = result_raw.copy()
        result_map['_race_key'] = result_map['賽事編號'].map(normalize_race_id)
        result_map['_horse_key'] = result_map['馬號'].map(normalize_horse_no)
        rename_map = {'名次': '_result_rank'}
        if '獨贏賠率' in result_map.columns: rename_map['獨贏賠率'] = '_result_win_odds'
        if '位置賠率' in result_map.columns: rename_map['位置賠率'] = '_result_place_odds'
        result_map = result_map.drop_duplicates(['_race_key', '_horse_key'], keep='last').rename(columns=rename_map)

        df_backtest['_race_key'] = df_backtest['賽事編號'].map(normalize_race_id)
        df_backtest['_horse_key'] = df_backtest['馬號'].map(normalize_horse_no)
        df_backtest = df_backtest.merge(result_map[['_race_key', '_horse_key'] + [c for c in rename_map.values() if c in result_map.columns]], on=['_race_key', '_horse_key'], how='left')
        df_backtest['名次'] = df_backtest['_result_rank']
        df_backtest['numeric_rank'] = pd.to_numeric(df_backtest['名次'], errors='coerce').fillna(99)
        if '_result_win_odds' in df_backtest.columns: df_backtest['回測獨贏賠率'] = pd.to_numeric(df_backtest['_result_win_odds'], errors='coerce')
        if '_result_place_odds' in df_backtest.columns: df_backtest['回測位置賠率'] = pd.to_numeric(df_backtest['_result_place_odds'], errors='coerce')
        df_backtest = df_backtest[df_backtest['_result_rank'].notna()].copy()
    except Exception as exc: st.sidebar.error(f"回測讀取失敗：{exc}")

tab1, tab2 = st.tabs(["🎯 預測推薦與單場凍結", "📈 歷史回測 (買時選馬 vs 最終派彩)"])

with tab1:
    col_v, col_d, col_r, col_t = st.columns([1.2, 1.8, 1.2, 1.8])
    venue_code = "HV" if col_v.selectbox("場地", ["HV (跑馬地)", "ST (沙田)"], index=1 if str(df_raw.get('racecourse_code',pd.Series([''])).iloc[0])=='ST' else 0).startswith("HV") else "ST"
    api_date = col_d.text_input("API 日期 (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
    races_available = sorted(int(x) for x in df_raw['賽事編號'].map(extract_race_no).dropna().unique())
    target_race = col_r.selectbox("操作場次", races_available) if races_available else col_r.number_input("場次", 1, 15, 1)
    
    auto_col, interval_col, button_col, lock_col = st.columns([1.3, 1.3, 2.0, 2.0])
    auto_refresh = auto_col.checkbox("自動更新未凍結場次")
    refresh_seconds = interval_col.selectbox("間隔(秒)", [10, 30, 60, 120], index=1, disabled=not auto_refresh)
    manual_refresh = button_col.button("🔄 立即更新", use_container_width=True)
    if lock_col.button("📌 鎖定早盤基準", use_container_width=True):
        st.session_state['baseline_odds_map'] = {f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}": float(r['獨贏賠率']) for _,r in df.iterrows()}
        st.session_state['baseline_locked_time'] = datetime.now(hk_timezone).strftime("%H:%M:%S")
        st.rerun()

    f_col1, f_col2, f_col3 = st.columns([2.2, 1.8, 2.6])
    if int(target_race) not in st.session_state['frozen_races']:
        if f_col1.button(f"🔒 凍結第 {target_race} 場賠率", use_container_width=True, type="primary"):
            st.session_state['frozen_races'][int(target_race)] = datetime.now(hk_timezone).strftime("%H:%M:%S")
            st.rerun()
    else:
        if f_col1.button(f"🔓 解除第 {target_race} 場凍結", use_container_width=True):
            st.session_state['frozen_races'].pop(int(target_race), None)
            st.rerun()
    if f_col2.button("🔓 解除全部凍結", use_container_width=True): st.session_state['frozen_races'] = {}; st.rerun()
    f_col3.info("🔒 已凍結: " + (", ".join([f"R{r}" for r in sorted(st.session_state['frozen_races'].keys())]) if st.session_state['frozen_races'] else "無"))

    if auto_refresh: st_autorefresh(interval=int(refresh_seconds * 1000), key="odds_refresh")
    if auto_refresh or manual_refresh:
        with st.spinner("讀取賠率中…"):
            live_by_race, _, _ = fetch_live_odds(api_date, venue_code)
        if live_by_race:
            df_t = st.session_state['df_data'].copy()
            for r_no, mkts in live_by_race.items():
                if int(r_no) in st.session_state['frozen_races']: continue
                rmask = df_t['賽事編號'].map(extract_race_no) == int(r_no)
                for h_no, v in mkts['WIN'].items(): df_t.loc[rmask & (df_t['馬號'].map(normalize_horse_no) == h_no), '獨贏賠率'] = v
                for h_no, v in mkts['PLA'].items(): df_t.loc[rmask & (df_t['馬號'].map(normalize_horse_no) == h_no), '位置賠率'] = v
            st.session_state['df_data'] = df_t
            st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version',0)+1
            st.rerun()

    recommendations = []
    for race_id, group in df.groupby('賽事編號'):
        sg = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds)
        if len(sg) > 0:
            t1 = sg.iloc[0]
            recommendations.append({
                '賽事編號': race_id,
                '賽道': t1.get('賽道', 'A'),
                '🎯 首選': f"{t1['馬號']}({t1['馬名']}) [勝率:{t1['pred_win_prob']*100:.1f}%|EV:{t1['ev']:.2f}] 💰${t1['suggested_stake']:.0f} {t1['資金流向']}",
                '🔗 Q/QP': f"{t1['馬號']}+{sg.iloc[1]['馬號']}" if len(sg)>1 else "無"
            })
    st.subheader("🎯 推薦清單")
    if recommendations: st.dataframe(pd.DataFrame(recommendations), hide_index=True)
    
    st.markdown(f"#### 🔍 第 {target_race} 場 深度雷達 (含檔位偏差)")
    rdf = df[df['賽事編號'].map(extract_race_no) == target_race].copy()
    if not rdf.empty:
        sort_col = 'smart_score' if strategy_mode.startswith("🛡️") else ('pred_win_prob' if strategy_mode.startswith("🎯") else 'ev')
        rdf = rdf.sort_values(by=sort_col, ascending=False)
        st.dataframe(pd.DataFrame({
            '馬號': rdf['馬號'], '馬名': rdf['馬名'], '檔位': rdf['排位檔位'], '賽道': rdf.get('賽道', 'A'),
            '檔位偏差調整': rdf['track_bias_adj'].map(lambda x: f"{x*100:+.0f}%" if x!=0 else "-"),
            'AI勝率': (rdf['pred_win_prob']*100).round(1).astype(str)+'%',
            '賠率': rdf['獨贏賠率'].round(1), '流向': rdf['資金流向'], 'EV': rdf['ev'].round(2),
            '注碼': rdf['suggested_stake'].apply(lambda x: f"${x:.0f}")
        }), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("📊 回測總覽")
    if df_backtest.empty or not backtest_ready: st.warning("請上傳完賽 CSV")
    else:
        st.caption(f"排序: `{strategy_mode}` | 門檻: EV>={min_ev}, 賠率 {min_odds}~{max_odds}")
        win_invested, win_return, win_hits = 0, 0, 0
        recs = []
        for rid, grp in df_backtest.groupby('賽事編號'):
            sg = filter_and_sort_group(grp, strategy_mode, min_ev, min_odds, max_odds)
            if len(sg) > 0:
                pk = sg.iloc[0]
                win_invested += 100
                sp = float(pk.get('回測獨贏賠率', pk['獨贏賠率'])) if pd.notna(pk.get('回測獨贏賠率')) else pk['獨贏賠率']
                if pk['numeric_rank'] == 1:
                    win_hits += 1; win_return += 100 * sp
                recs.append({'賽場': rid, '馬匹': f"{pk['馬號']}({pk['馬名']})", '賽道': pk.get('賽道','A'), '檔位': pk['排位檔位'], '結果': "✅贏" if pk['numeric_rank']==1 else "❌輸", '派彩': 100*sp if pk['numeric_rank']==1 else 0})
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("場數", len(recs))
        c2.metric("命中", win_hits, f"{win_hits/max(len(recs),1)*100:.1f}%")
        c3.metric("總成本", f"${win_invested}")
        c4.metric("總回收", f"${win_return:.1f}", f"ROI: {((win_return-win_invested)/max(win_invested,1))*100:.1f}%")
        if recs: st.dataframe(pd.DataFrame(recs), hide_index=True)