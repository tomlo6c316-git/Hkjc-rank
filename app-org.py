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
import json
import sqlite3
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

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統 (雙渦輪引擎+上仗同程+步速+手風+賽道+尋優)")
st.markdown("---")

# 初始化模擬投注紀錄與單場凍結名單
if 'simulated_bets' not in st.session_state:
    st.session_state['simulated_bets'] = pd.DataFrame(columns=[
        '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
    ])
if 'frozen_races' not in st.session_state:
    st.session_state['frozen_races'] = {}  # 格式: {race_no: "HH:MM:SS"}
if 'odds_monitor_history' not in st.session_state:
    st.session_state['odds_monitor_history'] = []
if 'odds_server_times_by_race' not in st.session_state:
    st.session_state['odds_server_times_by_race'] = {}
if 'odds_last_checked_by_race' not in st.session_state:
    st.session_state['odds_last_checked_by_race'] = {}
if 'place_baseline_odds_map' not in st.session_state:
    st.session_state['place_baseline_odds_map'] = {}

# 設定 repo 內的模型及預測 CSV 路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'
PLACE_MODEL_PATH = APP_DIR / 'my_hkjc_place_model.pkl'
PREDICTION_CSV_PATH = APP_DIR / 'prediction.csv'
FEATURE_COLS_15 = [
    'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
    '排位檔位', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    'horse_win_rate', 'horse_last_rank',
    '距離', 'horse_surface_win_rate', 'horse_dist_win_rate',
]

AUDIT_DB_PATH = Path(os.environ.get('HKJC_AUDIT_DB_PATH', str(Path.home() / '.hkjc_audit.sqlite3')))
AUDIT_CONTEXT_FIELDS = [
    '馬名', '騎師', '練馬師', '實際負磅', '排位檔位', '距離', '場地', '賽道', '跑法',
    '獨贏賠率', '位置賠率', 'pred_win_prob', 'pred_place_prob', 'odds_rank',
    'horse_last_rank', 'horse_surface_win_rate', 'horse_dist_win_rate',
    'market_implied_prob', 'is_favorite', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate', 'horse_win_rate',
    'snapshot_horse_last_rank', 'snapshot_horse_surface_win_rate', 'snapshot_horse_dist_win_rate',
    'snapshot_pred_win_prob', 'snapshot_pred_place_prob', 'snapshot_place_rank',
    'snapshot_model_form', 'snapshot_asof_date', 'snapshot_quote_source',
    'snapshot_win_model_signature', 'snapshot_place_model_signature',
]

def get_supabase_config():
    """Read cloud database credentials from Streamlit secrets or environment only."""
    url = os.environ.get('SUPABASE_URL', '')
    key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
    try:
        url = url or str(st.secrets.get('SUPABASE_URL', ''))
        key = key or str(st.secrets.get('SUPABASE_SERVICE_ROLE_KEY', ''))
    except Exception:
        pass
    return url.rstrip('/'), key

def supabase_request(table, method='POST', payload=None, params=None, on_conflict=None):
    """Best-effort Supabase PostgREST call; never interrupt live odds polling."""
    url, key = get_supabase_config()
    if not url or not key:
        return None
    headers = {
        'apikey': key,
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates,return=minimal',
    }
    query = dict(params or {})
    if on_conflict:
        query['on_conflict'] = on_conflict
    try:
        endpoint = f'{url}/rest/v1/{table}'
        if method == 'GET':
            response = requests.get(endpoint, headers=headers, params=query, timeout=12)
        else:
            response = requests.post(endpoint, headers=headers, params=query, json=payload, timeout=12)
        response.raise_for_status()
        return response.json() if method == 'GET' and response.text.strip() else True
    except Exception as exc:
        st.session_state['audit_cloud_error'] = f'{table}: {type(exc).__name__}: {exc}'
        return None

def supabase_get_all(table, params=None, page_size=1000, max_pages=100):
    """Fetch paginated PostgREST rows, respecting project row caps."""
    all_rows = []
    base_params = dict(params or {})
    for page in range(int(max_pages)):
        page_params = {**base_params, 'limit': str(int(page_size)), 'offset': str(page * int(page_size))}
        result = supabase_request(table, method='GET', params=page_params)
        if not isinstance(result, list) or not result:
            break
        all_rows.extend(result)
        if len(result) < int(page_size):
            break
    return all_rows

def init_audit_db():
    """Create local durable tables; cloud persistence is enabled when Supabase secrets exist."""
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''CREATE TABLE IF NOT EXISTS odds_snapshots (
            id TEXT PRIMARY KEY, race_id TEXT NOT NULL, captured_at TEXT NOT NULL,
            race_date TEXT, venue TEXT, race_no INTEGER, horse_no TEXT, horse_name TEXT,
            win_odds REAL, place_odds REAL, win_baseline REAL, place_baseline REAL,
            win_drop REAL, place_drop REAL, win_pool_updated_at TEXT, place_pool_updated_at TEXT,
            context_json TEXT, cloud_synced INTEGER NOT NULL DEFAULT 0)''')
        snapshot_columns = {row[1] for row in conn.execute('PRAGMA table_info(odds_snapshots)').fetchall()}
        if 'cloud_synced' not in snapshot_columns:
            conn.execute('ALTER TABLE odds_snapshots ADD COLUMN cloud_synced INTEGER NOT NULL DEFAULT 0')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_odds_snapshots_race ON odds_snapshots(race_id, captured_at)')
        conn.execute('''CREATE TABLE IF NOT EXISTS race_entry_context (
            race_id TEXT NOT NULL, horse_no TEXT NOT NULL, captured_at TEXT NOT NULL,
            context_json TEXT NOT NULL, PRIMARY KEY(race_id, horse_no))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS race_postrace_reports (
            race_id TEXT PRIMARY KEY, completed_at TEXT NOT NULL, venue TEXT,
            race_no INTEGER, report_json TEXT NOT NULL, source_url TEXT)''')

def persist_odds_snapshots(rows):
    if not rows:
        return
    init_audit_db()
    cloud_rows = []
    db_rows = []
    for row in rows:
        race_id = normalize_race_id(f"{str(row.get('賽日', '')).replace('-', '')}-{int(row.get('場次', 0)):02d}")
        horse_no = normalize_horse_no(row.get('馬號', ''))
        captured_at = str(row.get('查詢時間(HKT)', ''))
        row_id = hashlib.sha256(f'{race_id}|{horse_no}|{captured_at}'.encode('utf-8')).hexdigest()
        context = {field: row.get(field) for field in AUDIT_CONTEXT_FIELDS if field in row}
        values = (
            row_id, race_id, captured_at, str(row.get('賽日', '')), str(row.get('場地', '')),
            int(row.get('場次', 0)), horse_no, str(row.get('馬名', '')),
            _to_float_or_none(row.get('WIN賠率')), _to_float_or_none(row.get('PLACE賠率')),
            _to_float_or_none(row.get('WIN基準賠率')), _to_float_or_none(row.get('PLACE基準賠率')),
            _to_float_or_none(row.get('WIN落飛百分比')), _to_float_or_none(row.get('PLACE落飛百分比')),
            str(row.get('HKJC_WIN_LastUpdate') or ''), str(row.get('HKJC_PLACE_LastUpdate') or ''),
            json.dumps(context, ensure_ascii=False, default=str),
        )
        db_rows.append(values)
        cloud_rows.append({
            'id': row_id, 'race_id': race_id, 'captured_at': captured_at,
            'race_date': str(row.get('賽日', '')), 'venue': str(row.get('場地', '')),
            'race_no': int(row.get('場次', 0)), 'horse_no': horse_no,
            'horse_name': str(row.get('馬名', '')),
            'win_odds': values[8], 'place_odds': values[9],
            'win_baseline': values[10], 'place_baseline': values[11],
            'win_drop': values[12], 'place_drop': values[13],
            'win_pool_updated_at': values[14], 'place_pool_updated_at': values[15],
            'context': context,
        })
    with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
        conn.executemany('''INSERT OR IGNORE INTO odds_snapshots
            (id, race_id, captured_at, race_date, venue, race_no, horse_no, horse_name,
             win_odds, place_odds, win_baseline, place_baseline, win_drop, place_drop,
             win_pool_updated_at, place_pool_updated_at, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', db_rows)
    if get_supabase_config()[0]:
        sync_pending_snapshots()

def sync_pending_snapshots(limit=500):
    """Retry locally persisted snapshots not yet acknowledged by the cloud database."""
    if not get_supabase_config()[0]:
        return 0
    with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
        rows = conn.execute('''SELECT id, race_id, captured_at, race_date, venue, race_no,
            horse_no, horse_name, win_odds, place_odds, win_baseline, place_baseline,
            win_drop, place_drop, win_pool_updated_at, place_pool_updated_at, context_json
            FROM odds_snapshots WHERE cloud_synced=0 ORDER BY captured_at LIMIT ?''', (int(limit),)).fetchall()
    if not rows:
        return 0
    payload = []
    for row in rows:
        try:
            context = json.loads(row[16] or '{}')
        except Exception:
            context = {}
        payload.append({
            'id': row[0], 'race_id': row[1], 'captured_at': row[2], 'race_date': row[3],
            'venue': row[4], 'race_no': row[5], 'horse_no': row[6], 'horse_name': row[7],
            'win_odds': row[8], 'place_odds': row[9], 'win_baseline': row[10],
            'place_baseline': row[11], 'win_drop': row[12], 'place_drop': row[13],
            'win_pool_updated_at': row[14], 'place_pool_updated_at': row[15], 'context': context,
        })
    if supabase_request('odds_monitor_snapshots', payload=payload, on_conflict='id'):
        with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
            conn.executemany('UPDATE odds_snapshots SET cloud_synced=1 WHERE id=?', [(row[0],) for row in rows])
        return len(rows)
    return 0

def _to_float_or_none(value):
    parsed = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    return float(parsed) if pd.notna(parsed) and np.isfinite(parsed) else None

def persist_race_entry_context(df):
    """Freeze the first available pre-race model/card context per runner for later comparison."""
    if not isinstance(df, pd.DataFrame) or df.empty or '賽事編號' not in df.columns:
        return
    init_audit_db()
    captured_at = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S%z')
    local_rows, cloud_rows = [], []
    for _, row in df.iterrows():
        race_id = normalize_race_id(row.get('賽事編號', ''))
        horse_no = normalize_horse_no(row.get('馬號', ''))
        context = {field: row.get(field) for field in AUDIT_CONTEXT_FIELDS if field in df.columns}
        payload = json.dumps(context, ensure_ascii=False, default=str)
        local_rows.append((race_id, horse_no, captured_at, payload))
        cloud_rows.append({'race_id': race_id, 'horse_no': horse_no, 'captured_at': captured_at, 'context': context})
    stable_rows = []
    dynamic_fields = {'獨贏賠率', '位置賠率', 'odds_rank', 'pred_win_prob', 'pred_place_prob'}
    for item in cloud_rows:
        stable_rows.append({
            'race_id': item['race_id'], 'horse_no': item['horse_no'],
            'context': {k: v for k, v in item['context'].items() if k not in dynamic_fields},
        })
    signature = hashlib.sha256(json.dumps(stable_rows, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')).hexdigest()
    if st.session_state.get('audit_context_signature') != signature:
        with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
            conn.executemany('''INSERT OR IGNORE INTO race_entry_context(race_id, horse_no, captured_at, context_json)
                VALUES (?, ?, ?, ?)''', local_rows)
        st.session_state['audit_context_signature'] = signature
    if get_supabase_config()[0] and st.session_state.get('audit_context_cloud_signature') != signature:
        with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
            stored_contexts = []
            for race_id, horse_no, _, _ in local_rows:
                stored = conn.execute('SELECT captured_at, context_json FROM race_entry_context WHERE race_id=? AND horse_no=?', (race_id, horse_no)).fetchone()
                if stored:
                    stored_contexts.append({'race_id': race_id, 'horse_no': horse_no, 'captured_at': stored[0], 'context': json.loads(stored[1] or '{}')})
        if supabase_request('race_entry_context', payload=stored_contexts, on_conflict='race_id,horse_no'):
            st.session_state['audit_context_cloud_signature'] = signature

def load_race_audit_data(race_id):
    init_audit_db()
    with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
        snapshots = pd.read_sql_query('SELECT * FROM odds_snapshots WHERE race_id=? ORDER BY captured_at', conn, params=(race_id,))
        contexts = pd.read_sql_query('SELECT * FROM race_entry_context WHERE race_id=?', conn, params=(race_id,))
    if get_supabase_config()[0]:
        remote = supabase_get_all('odds_monitor_snapshots', params={
            'race_id': f'eq.{race_id}', 'order': 'captured_at.asc'
        })
        if isinstance(remote, list) and remote:
            remote_df = pd.DataFrame(remote)
            if 'context' in remote_df:
                remote_df['context_json'] = remote_df['context'].apply(lambda v: json.dumps(v or {}, ensure_ascii=False))
            snapshots = pd.concat([snapshots, remote_df], ignore_index=True, sort=False)
            if 'id' in snapshots:
                snapshots = snapshots.drop_duplicates('id', keep='last').sort_values('captured_at')
    if get_supabase_config()[0]:
        remote_contexts = supabase_get_all('race_entry_context', params={
            'race_id': f'eq.{race_id}'
        })
        if isinstance(remote_contexts, list) and remote_contexts:
            remote_contexts_df = pd.DataFrame(remote_contexts)
            if 'context' in remote_contexts_df:
                remote_contexts_df['context_json'] = remote_contexts_df['context'].apply(lambda v: json.dumps(v or {}, ensure_ascii=False))
            contexts = pd.concat([contexts, remote_contexts_df], ignore_index=True, sort=False)
            if 'horse_no' in contexts:
                contexts = contexts.drop_duplicates('horse_no', keep='last')
    if not snapshots.empty and 'context_json' in snapshots.columns:
        def parse_snapshot_context(value):
            if isinstance(value, dict):
                return value
            try:
                return json.loads(value or '{}')
            except (TypeError, ValueError):
                return {}
        expanded = snapshots['context_json'].apply(parse_snapshot_context)
        context_keys = sorted({key for item in expanded for key in item})
        for key in context_keys:
            if key not in snapshots.columns:
                snapshots[key] = expanded.apply(lambda item: item.get(key))
    return snapshots, contexts

def save_postrace_report(race_id, venue, race_no, completed_at, report_df, source_url):
    init_audit_db()
    report_json = report_df.to_json(orient='records', force_ascii=False)
    with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
        conn.execute('''INSERT INTO race_postrace_reports(race_id, completed_at, venue, race_no, report_json, source_url)
            VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(race_id) DO UPDATE SET
            completed_at=excluded.completed_at, venue=excluded.venue, race_no=excluded.race_no,
            report_json=excluded.report_json, source_url=excluded.source_url''',
            (race_id, completed_at, venue, int(race_no), report_json, source_url))
    if get_supabase_config()[0]:
        supabase_request('race_postrace_reports', payload=[{
            'race_id': race_id, 'completed_at': completed_at, 'venue': venue,
            'race_no': int(race_no), 'report': json.loads(report_json), 'source_url': source_url
        }], on_conflict='race_id')

def load_postrace_report(race_id):
    init_audit_db()
    with sqlite3.connect(AUDIT_DB_PATH, timeout=15) as conn:
        row = conn.execute('SELECT completed_at, report_json, source_url FROM race_postrace_reports WHERE race_id=?', (race_id,)).fetchone()
    if row:
        return pd.read_json(StringIO(row[1]), orient='records'), row[0], row[2]
    if get_supabase_config()[0]:
        remote = supabase_request('race_postrace_reports', method='GET', params={'race_id': f'eq.{race_id}', 'limit': '1'})
        if isinstance(remote, list) and remote:
            report = remote[0].get('report', [])
            return pd.DataFrame(report), remote[0].get('completed_at', ''), remote[0].get('source_url', '')
    return pd.DataFrame(), '', ''

def fetch_official_race_results(date_value, venue, race_no):
    """Read HKJC LocalResults and return official finishing order for one race, or empty when unavailable."""
    date_text = pd.Timestamp(date_value).strftime('%Y/%m/%d')
    url = f'https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={date_text}&RaceNo={int(race_no)}'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=18)
    response.encoding = 'utf-8'
    if '沒有相關資料' in response.text or 'No information' in response.text:
        return pd.DataFrame(), url
    try:
        tables = pd.read_html(StringIO(response.text))
    except Exception:
        return pd.DataFrame(), url
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = ['_'.join(map(str, col)) for col in table.columns]
        table.columns = [str(c).replace('\n', ' ').strip() for c in table.columns]
        header_text = ' '.join(table.columns) + ' ' + ' '.join(' '.join(map(str, table.iloc[i].tolist())) for i in range(min(5, len(table))))
        if '名次' not in header_text or not any(term in header_text for term in ('馬號', '編號')):
            continue
        if not any('名次' in c for c in table.columns) or not any(('馬號' in c or '編號' in c) for c in table.columns):
            for header_row in range(min(5, len(table))):
                row_headers = [str(value).replace('\n', ' ').strip() for value in table.iloc[header_row].tolist()]
                row_text = ' '.join(row_headers)
                if '名次' in row_text and any(term in row_text for term in ('馬號', '編號')):
                    table.columns = row_headers
                    table = table.iloc[header_row + 1:].reset_index(drop=True)
                    break
        rank_col = find_column(table.columns, ('名次',))
        horse_no_col = find_column(table.columns, ('馬號', '編號'))
        if not rank_col or not horse_no_col:
            continue
        horse_name_col = find_column(table.columns, ('馬名',))
        out = []
        for _, row in table.iterrows():
            horse_no = normalize_horse_no(row.get(horse_no_col, ''))
            rank_text = str(row.get(rank_col, '')).strip()
            match = re.match(r'^(\d+)', rank_text)
            if not horse_no.isdigit() or not match:
                continue
            item = {'馬號': horse_no, '官方名次': int(match.group(1)), '馬名': str(row.get(horse_name_col, '')).strip()}
            for key_words, target in [(('騎師',), '官方騎師'), (('練馬師',), '官方練馬師'), (('獨贏',), '賽後獨贏賠率'), (('沿途', '走位'), '沿途走位'), (('分段', '時間'), '分段時間'), (('勝負', '距離'), '勝負距離')]:
                col = find_column(table.columns, key_words)
                if col:
                    item[target] = str(row.get(col, '')).strip()
            out.append(item)
        if out:
            return pd.DataFrame(out).drop_duplicates('馬號', keep='first'), url
    return pd.DataFrame(), url

def make_postrace_report(date_value, venue, race_no, results_df, source_url):
    race_id = normalize_race_id(f'{pd.Timestamp(date_value):%Y%m%d}-{int(race_no):02d}')
    snapshots, contexts = load_race_audit_data(race_id)
    if results_df.empty:
        return pd.DataFrame(), race_id
    top3 = results_df[pd.to_numeric(results_df['官方名次'], errors='coerce').le(3)].copy().sort_values('官方名次')
    if top3.empty:
        return pd.DataFrame(), race_id
    for col in ['horse_no', '馬號']:
        if col in snapshots.columns:
            snapshots['馬號'] = snapshots[col].astype(str).map(normalize_horse_no)
            break
    context_map = {}
    if not contexts.empty:
        for _, context_row in contexts.iterrows():
            try:
                context_map[normalize_horse_no(context_row.get('horse_no', context_row.get('馬號', '')))] = json.loads(context_row.get('context_json', '{}') or '{}')
            except Exception:
                pass
    rows = []
    for _, result in top3.iterrows():
        horse_no = normalize_horse_no(result.get('馬號', ''))
        horse_snaps = snapshots[snapshots['馬號'].astype(str).map(normalize_horse_no).eq(horse_no)].copy() if not snapshots.empty and '馬號' in snapshots.columns else pd.DataFrame()
        context = context_map.get(horse_no, {})
        row = {
            '官方名次': int(result['官方名次']), '馬號': horse_no,
            '馬名': result.get('馬名') or context.get('馬名', ''),
            '監測快照數': len(horse_snaps),
        }
        for source, label in [('win_odds', '首個WIN報價'), ('place_odds', '首個PLACE報價'), ('WIN賠率', '首個WIN報價'), ('PLACE賠率', '首個PLACE報價')]:
            if source in horse_snaps.columns:
                values = pd.to_numeric(horse_snaps[source], errors='coerce').dropna()
                if not values.empty:
                    row[label] = round(float(values.iloc[0]), 2)
        mappings = [
            ('win_drop', 'WIN最大落飛%'), ('place_drop', 'PLACE最大落飛%'),
            ('WIN落飛百分比', 'WIN最大落飛%'), ('PLACE落飛百分比', 'PLACE最大落飛%'),
        ]
        for source, label in mappings:
            if source in horse_snaps.columns:
                values = pd.to_numeric(horse_snaps[source], errors='coerce').dropna()
                if not values.empty:
                    row[label] = round(float(values.max()), 2)
                    time_col = 'captured_at' if 'captured_at' in horse_snaps.columns else '查詢時間(HKT)'
                    if time_col in horse_snaps.columns:
                        peak_index = values.idxmax()
                        row['WIN最大落飛時間' if 'win' in source.lower() or 'WIN' in source else 'PLACE最大落飛時間'] = horse_snaps.loc[peak_index, time_col]
        for source, label in [('win_odds', '最後WIN報價'), ('place_odds', '最後PLACE報價'), ('WIN賠率', '最後WIN報價'), ('PLACE賠率', '最後PLACE報價')]:
            if source in horse_snaps.columns:
                values = pd.to_numeric(horse_snaps[source], errors='coerce').dropna()
                if not values.empty:
                    row[label] = round(float(values.iloc[-1]), 2)
        for field in ['pred_win_prob', 'pred_place_prob', '排位檔位', '實際負磅', '騎師', '練馬師', '距離', '場地', '賽道', '跑法', 'horse_last_rank', 'horse_surface_win_rate', 'horse_dist_win_rate']:
            if field in context:
                value = context[field]
                if field in ('pred_win_prob', 'pred_place_prob'):
                    value = round(float(value) * 100, 2) if _to_float_or_none(value) is not None else value
                label = {'pred_win_prob': 'AI勝率估計(%)', 'pred_place_prob': 'AI入位率估計(%)'}.get(field, field)
                row[label] = value
        for col in ['沿途走位', '分段時間', '勝負距離', '賽後獨贏賠率']:
            if col in result and result.get(col):
                row[col] = result[col]
        observations = []
        win_drop = _to_float_or_none(row.get('WIN最大落飛%'))
        place_drop = _to_float_or_none(row.get('PLACE最大落飛%'))
        if win_drop is not None:
            observations.append(f"WIN最大落飛{win_drop:.1f}%" if win_drop > 0 else f"WIN未見淨落飛（{win_drop:.1f}%）")
        if place_drop is not None:
            observations.append(f"PLACE最大落飛{place_drop:.1f}%" if place_drop > 0 else f"PLACE未見淨落飛（{place_drop:.1f}%）")
        ai_win = _to_float_or_none(row.get('AI勝率估計(%)'))
        ai_place = _to_float_or_none(row.get('AI入位率估計(%)'))
        if ai_win is not None:
            observations.append(f"賽前AI勝率估計{ai_win:.1f}%")
        if ai_place is not None:
            observations.append(f"賽前AI入位率估計{ai_place:.1f}%")
        row['觀察摘要'] = ('；'.join(observations) if observations else '沒有足夠賽前監測／模型紀錄') + '。只反映同期關聯，不能單獨證明取勝原因。'
        rows.append(row)
    report_df = pd.DataFrame(rows)
    completed_at = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S%z')
    save_postrace_report(race_id, venue, race_no, completed_at, report_df, source_url)
    return report_df, race_id

def auto_check_race_results(date_value, venue, race_numbers, prefer_race=None):
    """Throttle official-result checks and archive reports when HKJC publishes a result."""
    checked = st.session_state.setdefault('postrace_last_check', {})
    reports = st.session_state.setdefault('postrace_reports', {})
    now = time.time()
    race_numbers = sorted({int(r) for r in race_numbers if pd.notna(r)})
    if prefer_race in race_numbers:
        race_numbers.remove(int(prefer_race))
        race_numbers.insert(0, int(prefer_race))
    cursor = int(st.session_state.get('postrace_round_robin_cursor', 0))
    other_races = [r for r in race_numbers if r != prefer_race]
    if other_races and prefer_race is None:
        other_races = other_races[cursor % len(other_races):] + other_races[:cursor % len(other_races)]
    candidates = ([int(prefer_race)] if prefer_race in race_numbers else []) + [r for r in other_races if r != prefer_race]
    eligible = [r for r in candidates if now - float(checked.get(normalize_race_id(f'{pd.Timestamp(date_value):%Y%m%d}-{r:02d}'), 0)) >= 90][:1]
    st.session_state['postrace_round_robin_cursor'] = cursor + len(eligible)
    for race_no in eligible:
        race_no = int(race_no)
        race_id = normalize_race_id(f'{pd.Timestamp(date_value):%Y%m%d}-{race_no:02d}')
        checked[race_id] = now
        try:
            results, source_url = fetch_official_race_results(date_value, venue, race_no)
            if not results.empty:
                report_df, _ = make_postrace_report(date_value, venue, race_no, results, source_url)
                if not report_df.empty:
                    reports[race_id] = report_df.to_dict(orient='records')
        except Exception as exc:
            st.session_state['postrace_last_error'] = f'第{race_no}場官方賽果檢查失敗：{type(exc).__name__}: {exc}'

def _model_file_signature(model_path):
    """Return a cache key that changes when a model is added or replaced."""
    try:
        stat = Path(model_path).stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return "missing"

@st.cache_resource
def load_models(win_model_path, place_model_path, win_signature, place_signature):
    win_m = joblib.load(win_model_path) if win_signature != "missing" else None
    place_m = joblib.load(place_model_path) if place_signature != "missing" else None
    return win_m, place_m

model_win, model_place = load_models(
    str(MODEL_PATH),
    str(PLACE_MODEL_PATH),
    _model_file_signature(MODEL_PATH),
    _model_file_signature(PLACE_MODEL_PATH),
)

if model_win is None:
    st.error(f"⚠️ 找不到獨贏 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

if model_place is not None:
    st.sidebar.success("🧠 雙渦輪引擎已啟動：獨贏模型 + 位置模型")
else:
    st.sidebar.warning("⚠️ 找不到 PLACE 模型，位置彩池將暫用獨贏模型代算。")
    st.sidebar.caption(f"預期 PLACE 模型路徑：`{PLACE_MODEL_PATH}`")

@st.cache_data
def load_historical_stats(history_csv_path='hkjc_all_seasons_features.csv'):
    """從歷史大表中計算騎師、練馬師、馬匹勝率(含近50場手風)，以及馬匹上仗名次、場地勝率、路程勝率"""
    csv_path = Path(__file__).resolve().parent / history_csv_path
    if not os.path.exists(csv_path):
        return None
    
    try:
        try:
            hist_df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
        except UnicodeDecodeError:
            hist_df = pd.read_csv(csv_path, encoding='cp950', low_memory=False)
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
            
        recent_j_df = valid_ranks_df.groupby('_clean_jockey').tail(50)
        recent_jockey_stats = recent_j_df.groupby('_clean_jockey', as_index=False)['is_win'].mean().rename(columns={'is_win': 'recent_jockey_win_rate'})
        jockey_stats = jockey_stats.merge(recent_jockey_stats, on='_clean_jockey', how='left')

        recent_t_df = valid_ranks_df.groupby('_clean_trainer').tail(50)
        recent_trainer_stats = recent_t_df.groupby('_clean_trainer', as_index=False)['is_win'].mean().rename(columns={'is_win': 'recent_trainer_win_rate'})
        trainer_stats = trainer_stats.merge(recent_trainer_stats, on='_clean_trainer', how='left')

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
        if any(name in str(exc).lower() for name in ('lxml', 'html5lib', 'bs4', 'beautifulsoup')):
            raise RuntimeError(
                '缺少 pandas HTML 表格解析依賴。請確認 lxml、html5lib、beautifulsoup4 已安裝。'
            ) from exc
        raise
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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    session = requests.Session()
    rows = []
    race_venue = None
    first_race = None
    for venue_code in ('ST', 'HV'):
        url = (
            'https://racing.hkjc.com/racing/information/Chinese/Racing/'
            f'RaceCard.aspx?racedate={date_text}&Racecourse={venue_code}&RaceNo=1'
        )
        response = session.get(url, headers=headers, timeout=12)
        response.encoding = 'utf-8'
        first_race_table = find_racecard_table(response.text)
        if first_race_table is not None:
            race_venue = venue_code
            first_race = (first_race_table, response.text)
            break
    if first_race is None:
        raise ValueError('HKJC 暫未提供該日期的賽卡。請確認所選日期為香港賽馬日，且排位表已公布。')

    for race_no in range(1, 16):
        if race_no == 1:
            race_table, race_html = first_race
        else:
            url = (
                'https://racing.hkjc.com/racing/information/Chinese/Racing/'
                f'RaceCard.aspx?racedate={date_text}&Racecourse={race_venue}&RaceNo={race_no}'
            )
            response = session.get(url, headers=headers, timeout=12)
            response.encoding = 'utf-8'
            race_html = response.text
            race_table = find_racecard_table(race_html)
            if race_table is None:
                break

        distance_match = re.search(r'(\d{4})\s*米', race_html)
        distance = int(distance_match.group(1)) if distance_match else 1200
        surface = '泥地' if ('泥地' in race_html or '全天候' in race_html) else '草地'
        
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
            if re.fullmatch(r'\d+\.0', horse_no):
                horse_no = horse_no[:-2]
            if not horse_no.isdigit():
                continue
            horse_name = re.sub(r'\(.*?\)', '', str(row[horse_name_col])).strip()
            if not horse_name or horse_name.lower() == 'nan':
                continue
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

    if not rows:
        raise ValueError('HKJC 沒有回傳排位表。')

    result = pd.DataFrame(rows)
    return result.drop_duplicates(['賽事編號', '馬號'], keep='last').reset_index(drop=True)

st.sidebar.header("📂 預測資料")
available_csvs = sorted([f.name for f in APP_DIR.glob("prediction*.csv")])

if not available_csvs:
    st.sidebar.caption("⚠️ 找不到任何 prediction*.csv 檔案。")
    selected_csv_name = "prediction.csv"
else:
    selected_csv_name = st.sidebar.selectbox(
        "📜 選擇 Repo 內的預測賽卡", 
        available_csvs,
        key="repo_csv_selector"
    )

if st.session_state.get('active_csv_name') != selected_csv_name:
    st.session_state['active_csv_name'] = selected_csv_name
    st.session_state.pop('fetched_prediction_df', None)
    st.session_state.pop('df_data', None)
    st.session_state.pop('prediction_signature', None)
    st.session_state.pop('baseline_odds_map', None)
    st.session_state.pop('place_baseline_odds_map', None)
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
        st.session_state.pop('place_baseline_odds_map', None)
        st.session_state.pop('baseline_locked_time', None)
        st.session_state['frozen_races'] = {}
        st.sidebar.success(f'已載入 {len(fetched_card)} 匹馬。')
    except Exception as exc:
        st.sidebar.error(f'抓取排位失敗：{type(exc).__name__}: {exc}')

if st.session_state.get('fetched_prediction_df') is not None:
    csv_bytes_pred = st.session_state['fetched_prediction_df'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.sidebar.download_button(
        label="⬇️ 下載已抓取的預測 CSV",
        data=csv_bytes_pred,
        file_name=f"prediction_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

current_fetched_card = st.session_state.get('fetched_prediction_df')
if current_fetched_card is not None:
    if st.sidebar.button('🔄 放棄抓取資料，改用選單 CSV', use_container_width=True):
        st.session_state.pop('fetched_prediction_df', None)
        st.session_state.pop('df_data', None)
        st.session_state.pop('prediction_signature', None)
        st.session_state.pop('baseline_odds_map', None)
        st.session_state.pop('place_baseline_odds_map', None)
        st.session_state.pop('baseline_locked_time', None)
        st.session_state['frozen_races'] = {}
        st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
        st.rerun()

backtest_file = st.sidebar.file_uploader(
    "回測用：上傳已完成賽事 CSV",
    type=['csv'],
    key='backtest_results_upload'
)

st.sidebar.markdown("---")
st.sidebar.header("⚙ 投注策略與算分引擎")
strategy_mode = st.sidebar.radio(
    "1️⃣ 推薦排序核心邏輯",
    ["⚡ 傳統純 EV 排序 (高賠率優先)", "🛡️ 抗噪穩定 (含步速+手風+賽道+排除回飛)", "🎯 純 AI 勝率/入位率最高 (命中率優先)"],
    index=0
)
use_dynamic_form = st.sidebar.checkbox("2️⃣ 將「上仗名次與同程勝率」納入 AI 算分", value=False)
min_win_ev = st.sidebar.slider("🥇 獨贏最小期望值 (Win EV)", 0.0, 6.0, 2.5, 0.1)
min_place_ev = st.sidebar.slider("🥉 位置最小期望值 (Place EV)", 0.0, 6.0, 1.5, 0.1)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=25.0)

st.sidebar.markdown("---")
st.sidebar.header("💰 資金管理 (凱利注碼精算)")
total_bankroll = st.sidebar.number_input("💵 單日總本金預算 (HKD)", min_value=100, max_value=1000000, value=500, step=500)
kelly_fraction = st.sidebar.slider("🎛️ 凱利保守係數", 0.05, 1.0, 0.15, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🛠️ 歷史賽果 15 大特徵抓取器")
tool_date_choice = st.sidebar.date_input("選擇目標賽事日期", value=datetime.now().date(), key="tool_date_input")
tool_season = st.sidebar.text_input("輸入馬季 (例如 23/24 或 26/27)", value="26/27")

if st.sidebar.button("📥 抓取並產出完賽 CSV", use_container_width=True, key="run_tool_button"):
    date_str_tool = tool_date_choice.strftime('%Y/%m/%d')
    date_for_id_tool = tool_date_choice.strftime('%Y%m%d')
    
    all_races_data_tool = []
    headers_tool = {'User-Agent': 'Mozilla/5.0'}
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    success_count = 0
    total_horses = 0
    
    for race_no in range(1, 13):
        status_text.text(f"正在抓取第 {race_no} 場資料...")
        url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={date_str_tool}&RaceNo={race_no}"
        try:
            resp = requests.get(url, headers=headers_tool, timeout=15)
            resp.encoding = 'utf-8' 
            html_text = resp.text
            
            if "沒有相關資料" in html_text or "No information" in html_text:
                progress_bar.progress(race_no / 12)
                continue

            distance = 1200
            dist_match = re.search(r'(\d{4})\s*米', html_text)
            if dist_match:
                distance = int(dist_match.group(1))
            surface_type = "草地" if "草地" in html_text else "泥地"
            
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
                horse_name_col = next((col for col in cols if '馬名' in col), None)
                jockey_col = next((col for col in cols if '騎師' in col), None)
                trainer_col = next((col for col in cols if '練馬師' in col), None)
                odds_col = next((col for col in cols if '獨贏' in col), None)
                weight_col = next((col for col in cols if '負磅' in col or '實際負磅' in col), None)
                draw_col = next((col for col in cols if '檔位' in col), None)
                
                if rank_col and horse_no_col:
                    added_in_this_race = 0
                    for idx, row in target_df.iterrows():
                        rank = str(row.get(rank_col, '')).strip()
                        horse_no = str(row.get(horse_no_col, '')).strip()
                        if horse_no.endswith('.0'): horse_no = horse_no[:-2]
                        if horse_no.isdigit():
                            horse_name_raw = str(row.get(horse_name_col, ''))
                            horse_name = horse_name_raw.split('(')[0].strip() if pd.notna(horse_name_raw) else ""
                            odds_val = row[odds_col] if odds_col and pd.notna(row[odds_col]) else 0.0
                            actual_weight = str(row[weight_col]).strip() if weight_col and pd.notna(row[weight_col]) else '120'
                            draw_pos = str(row[draw_col]).strip() if draw_col and pd.notna(row[draw_col]) else '7'
                            
                            all_races_data_tool.append({
                                '馬季': tool_season,
                                '賽事編號': f"{date_for_id_tool}-{race_no:02d}",
                                '名次': rank,
                                '馬號': horse_no,
                                '馬名': horse_name,
                                '騎師': str(row.get(jockey_col, '')).strip(),
                                '練馬師': str(row.get(trainer_col, '')).strip(),
                                '實際負磅': actual_weight,
                                '排位檔位': draw_pos,
                                '獨贏賠率': odds_val,
                                '距離': distance,                
                                '場地': surface_type,
                                '賽道': course_track,
                                'horse_surface_win_rate': 0.08,   
                                'horse_dist_win_rate': 0.08
                            })
                            added_in_this_race += 1
                    if added_in_this_race > 0:
                        success_count += 1
                        total_horses += added_in_this_race
            progress_bar.progress(race_no / 12)
            time.sleep(1.0)
        except Exception as e:
            continue
            
    progress_bar.empty()
    status_text.empty()
    if not all_races_data_tool:
        st.sidebar.error("❌ 該日期找不到任何賽事資料，或賽事已取消。")
    else:
        df_tool = pd.DataFrame(all_races_data_tool)
        df_tool['獨贏賠率'] = pd.to_numeric(df_tool['獨贏賠率'], errors='coerce').fillna(0.0)
        output_cols = ['馬季', '賽事編號', '名次', '馬號', '馬名', '騎師', '練馬師', '實際負磅', '排位檔位', '獨贏賠率', '距離', '場地', '賽道', 'horse_surface_win_rate', 'horse_dist_win_rate']
        df_tool = df_tool[[col for col in output_cols if col in df_tool.columns]]
        csv_filename = f"hkjc_data_15features_{date_for_id_tool}.csv"
        csv_data = df_tool.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.sidebar.success(f"🎉 成功抓取 {success_count} 場，共 {total_horses} 匹真實賽果資料！")
        st.sidebar.download_button(
            label="⬇️ 下載產出的完賽 CSV 檔",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )

st.sidebar.markdown("---")
st.sidebar.header("🔄 歷史大表合併工具")
new_csv_files = st.sidebar.file_uploader(
    "上傳新完賽 CSV (可多選)", 
    type=['csv'], 
    accept_multiple_files=True,
    key="merge_csv_uploader"
)
if new_csv_files and st.sidebar.button("⚡ 合併並產出最新歷史大表", use_container_width=True):
    hist_path = APP_DIR / 'hkjc_all_seasons_features.csv'
    dfs_to_merge = []
    if hist_path.exists():
        try:
            base_df = pd.read_csv(hist_path, encoding='utf-8-sig', low_memory=False)
        except Exception:
            base_df = pd.read_csv(hist_path, encoding='cp950', low_memory=False)
        dfs_to_merge.append(base_df)
        old_len = len(base_df)
    else:
        old_len = 0
        st.sidebar.warning("⚠️ 找不到現有大表，將僅合併上傳的檔案。")
        
    for f in new_csv_files:
        try:
            sub_df = pd.read_csv(f, encoding='utf-8-sig')
        except Exception:
            f.seek(0)
            sub_df = pd.read_csv(f, encoding='cp950')
        dfs_to_merge.append(sub_df)
        
    if dfs_to_merge:
        merged_df = pd.concat(dfs_to_merge, ignore_index=True)
        if '賽事編號' in merged_df.columns and '馬號' in merged_df.columns:
            merged_df['_r_key'] = merged_df['賽事編號'].astype(str).str.strip()
            merged_df['_h_key'] = merged_df['馬號'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            merged_df = merged_df.drop_duplicates(subset=['_r_key', '_h_key'], keep='last')
            merged_df = merged_df.sort_values(by=['_r_key']).drop(columns=['_r_key', '_h_key']).reset_index(drop=True)
            
        new_len = len(merged_df)
        added_rows = new_len - old_len
        merged_bytes = merged_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.sidebar.success(f"✅ 合併完成！原 {old_len:,} 筆 ➔ 現 {new_len:,} 筆 (淨新增 {added_rows:,} 筆)")
        st.sidebar.download_button(
            label="⬇️ 下載最新 hkjc_all_seasons_features.csv",
            data=merged_bytes,
            file_name="hkjc_all_seasons_features.csv",
            mime="text/csv",
            use_container_width=True
        )

GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"
ODDS_QUERY = """query racing($date: String,$venueCode: String, $oddsTypes: [OddsType],$raceNo: Int) {
  raceMeetings(date: $date, venueCode:$venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo:$raceNo) {
      id
      status
      sellStatus
      oddsType
      lastUpdateTime
      guarantee
      minTicketCost
      name_en
      name_ch
      leg {
        number
        races
      }
      cWinSelections {
        composite
        name_ch
        name_en
        starters
      }
      oddsNodes {
        combString
        oddsValue
        hotFavourite
        oddsDropValue
        bankerOdds {
          combString
          oddsValue
        }
      }
    }
  }
}"""

GRAPHQL_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Origin": "https://bet.hkjc.com",
    "Referer": "https://bet.hkjc.com/",
    "User-Agent": "Mozilla/5.0",
}

def normalize_horse_no(value):
    if pd.isna(value): return ""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (ValueError, TypeError):
        return text

def extract_race_no(value):
    match = re.search(r"-(\d+)\s*$", str(value).strip())
    return int(match.group(1)) if match else None

def normalize_race_id(value):
    text = str(value).strip()
    match = re.search(r"(20\d{6}).*?(\d+)\s*$", text)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    return text

def fetch_live_odds(date_str, venue, race_no=None):
    body = {
        "operationName": "racing",
        "variables": {
            "date": date_str,
            "venueCode": venue,
            "raceNo": int(race_no) if race_no is not None else None,
            "oddsTypes": ["WIN", "PLA"],
        },
        "query": ODDS_QUERY,
    }
    try:
        resp = requests.post(GRAPHQL_URL, headers=GRAPHQL_HEADERS, json=body, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            return None, None, f"GraphQL 錯誤：{payload['errors']}"

        meetings = (payload.get("data") or {}).get("raceMeetings") or []
        if not meetings:
            return None, None, "找不到該日期／場地的賽事資料。"

        odds_by_race = {}
        timestamps_by_race = {}
        for pool in meetings[0].get("pmPools") or []:
            pool_type = str(pool.get("oddsType", "")).upper()
            if pool_type not in ("WIN", "PLA"): continue

            race_numbers = (pool.get("leg") or {}).get("races") or []
            for race_number in race_numbers:
                race_number = int(race_number)
                if race_no is not None and race_number != int(race_no):
                    continue
                market = odds_by_race.setdefault(race_number, {"WIN": {}, "PLA": {}})
                timestamps_by_race.setdefault(race_number, {})[pool_type] = pool.get("lastUpdateTime")
                for node in pool.get("oddsNodes") or []:
                    horse_no = normalize_horse_no(node.get("combString"))
                    odds_value = str(node.get("oddsValue")).strip()
                    if horse_no and odds_value not in ("None", "", "SCR"):
                        try:
                            parsed_odds = float(odds_value)
                            if parsed_odds > 0:
                                market[pool_type][horse_no] = parsed_odds
                        except ValueError:
                            pass
        if not odds_by_race:
            return None, None, "賽事有回應，但目前尚無可用 WIN／PLACE 賠率。"
        return odds_by_race, timestamps_by_race, None
    except Exception as exc:
        return None, None, f"發生錯誤：{exc}"


def build_snapshot_model_context(base_df, live_by_race):
    """Re-score races with this poll's genuine WIN quotes; omit incomplete pools."""
    if not isinstance(base_df, pd.DataFrame) or base_df.empty:
        return {}
    scored = base_df.copy()
    race_numbers = scored['賽事編號'].map(extract_race_no)
    horse_numbers = scored['馬號'].map(normalize_horse_no)
    valid_races = set()
    for race_no, markets in (live_by_race or {}).items():
        race_mask = race_numbers.eq(int(race_no))
        quotes = markets.get('WIN') or {}
        all_valid = True
        for idx in scored.index[race_mask]:
            quote = quotes.get(horse_numbers.loc[idx])
            if quote is None or not np.isfinite(float(quote)) or float(quote) <= 1.0:
                all_valid = False
                break
            scored.at[idx, '獨贏賠率'] = float(quote)
        if all_valid and int(race_mask.sum()) >= 1:
            valid_races.add(int(race_no))

    context_by_race = {}
    for race_no in valid_races:
        mask = race_numbers.eq(race_no)
        race_df = scored.loc[mask].copy()
        odds = pd.to_numeric(race_df['獨贏賠率'], errors='coerce')
        raw_prob = 1.0 / odds
        race_df['market_implied_prob'] = raw_prob / raw_prob.sum()
        race_df['odds_rank'] = odds.rank(method='min', ascending=True)
        race_df['is_favorite'] = (race_df['odds_rank'] == 1).astype(int)
        X = race_df[FEATURE_COLS_15].replace([np.inf, -np.inf], np.nan).fillna(0)
        win_probs = model_win.predict_proba(X)[:, 1] if model_win is not None else [None] * len(X)
        place_probs = model_place.predict_proba(X)[:, 1] if model_place is not None else [None] * len(X)
        race_df['snapshot_pred_win_prob'] = win_probs
        race_df['snapshot_pred_place_prob'] = place_probs
        race_df['snapshot_place_rank'] = pd.Series(place_probs, index=race_df.index).rank(
            method='min', ascending=False
        ) if model_place is not None else np.nan
        form = 'Dynamic' if use_dynamic_form else 'Fixed'
        for idx, row in race_df.iterrows():
            horse_no = normalize_horse_no(row.get('馬號'))
            context = {column: _to_float_or_none(row.get(column)) for column in FEATURE_COLS_15}
            context.update({
                'snapshot_pred_win_prob': _to_float_or_none(row.get('snapshot_pred_win_prob')),
                'snapshot_pred_place_prob': _to_float_or_none(row.get('snapshot_pred_place_prob')),
                'snapshot_place_rank': _to_float_or_none(row.get('snapshot_place_rank')),
                'snapshot_horse_last_rank': _to_float_or_none(row.get('horse_last_rank')),
                'snapshot_horse_surface_win_rate': _to_float_or_none(row.get('horse_surface_win_rate')),
                'snapshot_horse_dist_win_rate': _to_float_or_none(row.get('horse_dist_win_rate')),
                'snapshot_model_form': form,
                'snapshot_asof_date': normalize_race_id(row.get('賽事編號', '')).split('-')[0],
                'snapshot_quote_source': 'HKJC GraphQL official WIN/PLACE pool response',
                'snapshot_win_model_signature': _model_file_signature(MODEL_PATH),
                'snapshot_place_model_signature': _model_file_signature(PLACE_MODEL_PATH),
            })
            context_by_race.setdefault(race_no, {})[horse_no] = context
    return context_by_race


def record_single_race_odds(date_str, venue, race_no, markets, server_times, snapshot_context_by_horse=None):
    """Keep an in-session view and persist every actual HKJC quote for one race."""
    # Microsecond precision prevents same-second manual polls from sharing the
    # SQLite/Supabase snapshot ID and makes each captured quote auditable.
    captured_at = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S.%f%z")
    times = (server_times or {}).get(int(race_no), {})
    horse_rows = {}
    current_df = st.session_state.get("df_data")
    if isinstance(current_df, pd.DataFrame) and not current_df.empty:
        race_nums = current_df["賽事編號"].map(extract_race_no)
        subset = current_df.loc[race_nums.eq(int(race_no))]
        for _, row in subset.iterrows():
            horse_rows[normalize_horse_no(row.get("馬號"))] = row

    win_quotes = (markets or {}).get("WIN", {})
    place_quotes = (markets or {}).get("PLA", {})
    horse_numbers = sorted(set(win_quotes) | set(place_quotes) | set(horse_rows), key=lambda x: (not str(x).isdigit(), str(x)))
    baseline_map = st.session_state.setdefault("baseline_odds_map", {})
    place_baseline_map = st.session_state.setdefault("place_baseline_odds_map", {})
    history = st.session_state.setdefault("odds_monitor_history", [])
    persisted_rows = []
    initialized_win_baseline = False
    for horse_no in horse_numbers:
        row = horse_rows.get(normalize_horse_no(horse_no))
        race_id = str(row.get("賽事編號", f"{date_str}-{int(race_no):02d}")) if row is not None else f"{date_str}-{int(race_no):02d}"
        horse_name = str(row.get("馬名", "")) if row is not None else ""
        key = f"{normalize_race_id(race_id)}_{normalize_horse_no(horse_no)}"
        baseline = pd.to_numeric(pd.Series([baseline_map.get(key)]), errors="coerce").iloc[0]
        place_baseline = pd.to_numeric(pd.Series([place_baseline_map.get(key)]), errors="coerce").iloc[0]
        win_value = win_quotes.get(normalize_horse_no(horse_no))
        place_value = place_quotes.get(normalize_horse_no(horse_no))
        # Start the drop baseline from the first real official quote seen for
        # this runner; never seed it from CSV defaults such as 10.0.
        if win_value is not None and float(win_value) > 1.0 and (pd.isna(baseline) or float(baseline) <= 1.0):
            baseline = float(win_value)
            baseline_map[key] = baseline
            initialized_win_baseline = True
        drop_pct = None
        place_drop_pct = None
        if win_value is not None and pd.notna(baseline) and float(baseline) > 0:
            drop_pct = round((float(baseline) - float(win_value)) / float(baseline) * 100.0, 3)
        if place_value is not None and float(place_value) > 0:
            if pd.isna(place_baseline) or float(place_baseline) <= 0:
                place_baseline = float(place_value)
                place_baseline_map[key] = float(place_value)
            place_drop_pct = round((float(place_baseline) - float(place_value)) / float(place_baseline) * 100.0, 3)
        snapshot = {
            "查詢時間(HKT)": captured_at,
            "賽日": date_str,
            "場地": venue,
            "場次": int(race_no),
            "馬號": normalize_horse_no(horse_no),
            "馬名": horse_name,
            "WIN賠率": win_value,
            "WIN基準賠率": float(baseline) if pd.notna(baseline) else None,
            "PLACE賠率": place_value,
            "PLACE基準賠率": float(place_baseline) if pd.notna(place_baseline) else None,
            "WIN落飛百分比": drop_pct,
            "PLACE落飛百分比": place_drop_pct,
            "HKJC_WIN_LastUpdate": times.get("WIN"),
            "HKJC_PLACE_LastUpdate": times.get("PLA"),
        }
        model_context = (snapshot_context_by_horse or {}).get(normalize_horse_no(horse_no), {})
        snapshot.update(model_context)
        history.append(snapshot)
        persisted_rows.append(snapshot)
    if initialized_win_baseline and st.session_state.get('baseline_locked_time') in (
        None, '', '初始載入值', '等候首個HKJC WIN實報'
    ):
        st.session_state['baseline_locked_time'] = f"首個HKJC WIN實報 ({captured_at})"
    try:
        persist_odds_snapshots(persisted_rows)
    except Exception as exc:
        st.session_state['audit_local_error'] = f'{type(exc).__name__}: {exc}'
    if len(history) > 20000:
        del history[:-20000]
    return captured_at, times


# 🎲 蒙地卡羅模擬引擎
def run_monte_carlo_exotics(race_detail_df, num_sims=50000, place_weight=0.5, random_seed=20261025):
    """以WIN與PLACE模型的相對強度混合後，按Plackett-Luce排序抽樣。

    PLACE機率在此只作「相對入位傾向」訊號：兩組機率各自正規化至總和1，
    再依place_weight混合。輸出的倒數機率是模型估算公平賠率，不是HKJC派彩。
    Gumbel top-k 等價於從該混合強度生成Plackett-Luce名次排序。
    """
    required = {'pred_win_prob', 'pred_place_prob', '馬號', '馬名'}
    missing = required - set(race_detail_df.columns)
    if missing:
        raise ValueError(f"蒙地卡羅缺少模型欄位：{sorted(missing)}")
    if not 0.0 <= float(place_weight) <= 1.0:
        raise ValueError('位置模型權重必須介乎0至1。')
    if int(num_sims) < 1000:
        raise ValueError('模擬次數至少需1,000次。')

    valid_horses = race_detail_df.copy()
    valid_horses['_win_signal'] = pd.to_numeric(valid_horses['pred_win_prob'], errors='coerce')
    valid_horses['_place_signal'] = pd.to_numeric(valid_horses['pred_place_prob'], errors='coerce')
    valid_horses = valid_horses.replace([np.inf, -np.inf], np.nan)
    valid_horses = valid_horses.dropna(subset=['_win_signal', '_place_signal'])
    valid_horses = valid_horses[(valid_horses['_win_signal'] > 0) & (valid_horses['_place_signal'] >= 0)]
    if len(valid_horses) < 3:
        return None, None, None

    horses = np.array(valid_horses['馬號'].astype(str).tolist())
    horse_names = np.array(valid_horses['馬名'].astype(str).tolist())
    win_signal = valid_horses['_win_signal'].to_numpy(dtype=float, copy=True)
    place_signal = valid_horses['_place_signal'].to_numpy(dtype=float, copy=True)
    win_signal /= win_signal.sum()
    if place_signal.sum() <= 0:
        raise ValueError('PLACE模型沒有有效的正入位機率。')
    place_signal /= place_signal.sum()
    weight = float(place_weight)
    strengths = (1.0 - weight) * win_signal + weight * place_signal
    strengths = np.clip(strengths, 1e-12, None)
    strengths /= strengths.sum()

    rng = np.random.default_rng(int(random_seed))
    gumbel_noise = rng.gumbel(0, 1, size=(int(num_sims), len(horses)))
    scores = np.log(strengths) + gumbel_noise
    top3_idx = np.argsort(-scores, axis=1)[:, :3]
    top3_horses = horses[top3_idx]
    name_map = dict(zip(horses, horse_names))
    place_slots = 2 if 4 <= len(horses) <= 6 else (3 if len(horses) >= 7 else 0)

    # 連贏Q：模擬頭兩名，不計名次先後。
    q_combos = np.sort(top3_horses[:, :2], axis=1)
    q_df = pd.DataFrame(q_combos, columns=['H1', 'H2'])
    q_counts = q_df.groupby(['H1', 'H2']).size().reset_index(name='hits')
    q_counts['模型估算機率'] = q_counts['hits'] / int(num_sims)
    q_counts['模型估算公平賠率'] = 1.0 / q_counts['模型估算機率']
    q_counts['組合名稱'] = q_counts['H1'].map(name_map) + " + " + q_counts['H2'].map(name_map)
    q_counts = q_counts.sort_values('模型估算機率', ascending=False).head(10)

    # 位置Q：兩駒均須入位；HKJC本地4–6匹兩席，7匹以上三席。
    if place_slots >= 2:
        qp_pairs = [np.sort(top3_horses[:, pair], axis=1)
                    for pair in __import__('itertools').combinations(range(place_slots), 2)]
        qp_all = np.vstack(qp_pairs)
        qp_df = pd.DataFrame(qp_all, columns=['H1', 'H2'])
        qp_counts = qp_df.groupby(['H1', 'H2']).size().reset_index(name='hits')
        qp_counts['模型估算機率'] = qp_counts['hits'] / int(num_sims)
        qp_counts['模型估算公平賠率'] = 1.0 / qp_counts['模型估算機率']
        qp_counts['組合名稱'] = qp_counts['H1'].map(name_map) + " + " + qp_counts['H2'].map(name_map)
        qp_counts = qp_counts.sort_values('模型估算機率', ascending=False).head(10)
    else:
        qp_counts = pd.DataFrame(columns=['H1', 'H2', 'hits', '模型估算機率', '模型估算公平賠率', '組合名稱'])

    # 單T：不計名次先後，三駒必須剛好佔據頭三名。
    trio = np.sort(top3_horses[:, :3], axis=1)
    trio_df = pd.DataFrame(trio, columns=['H1', 'H2', 'H3'])
    trio_counts = trio_df.groupby(['H1', 'H2', 'H3']).size().reset_index(name='hits')
    trio_counts['模型估算機率'] = trio_counts['hits'] / int(num_sims)
    trio_counts['模型估算公平賠率'] = 1.0 / trio_counts['模型估算機率']
    trio_counts['組合名稱'] = trio_counts['H1'].map(name_map) + " + " + trio_counts['H2'].map(name_map) + " + " + trio_counts['H3'].map(name_map)
    trio_counts = trio_counts.sort_values('模型估算機率', ascending=False).head(10)

    return q_counts, qp_counts, trio_counts


fetched_prediction_df = st.session_state.get('fetched_prediction_df')
hk_timezone = timezone(timedelta(hours=8))

if fetched_prediction_df is not None:
    df_raw = fetched_prediction_df.copy()
    fetched_date = st.session_state.get('fetched_prediction_date', race_date_choice)
    source_label = f"HKJC {fetched_date:%Y-%m-%d} 官方排位抓取"
    prediction_signature = st.session_state.get('fetched_prediction_signature', source_label)
    st.sidebar.success(f'預測來源：{source_label}（{len(df_raw)} 匹）')
elif SELECTED_CSV_PATH.is_file():
    try:
        df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='cp950')
    source_label = f"GitHub repo：{SELECTED_CSV_PATH.name}"
    prediction_signature = f"{SELECTED_CSV_PATH.name}:{SELECTED_CSV_PATH.stat().st_mtime_ns}:{SELECTED_CSV_PATH.stat().st_size}"
    if df_raw.empty:
        st.error(f"預測 CSV 是空檔：{SELECTED_CSV_PATH.name}")
        st.stop()
    st.sidebar.success(f"已從 repo 載入：{SELECTED_CSV_PATH.name}（{len(df_raw)} 匹）")
    st.success(f"✅ 目前顯示預測賽事檔案：`{SELECTED_CSV_PATH.name}`（共 {len(df_raw)} 匹馬）")
else:
    st.error(f"找不到 repo 預測 CSV：{SELECTED_CSV_PATH.name}。")
    st.stop()

auto_odds_status = None
if fetched_prediction_df is not None and st.session_state.pop('pending_auto_odds_fetch', False):
    odds_date = st.session_state.get('fetched_prediction_date', race_date_choice).strftime('%Y-%m-%d')
    odds_venue = str(df_raw['racecourse_code'].iloc[0])
    with st.spinner('排位已載入，正在嘗試取得該日 WIN／PLACE 賠率…'):
        odds_by_race, odds_times_by_race, odds_error = fetch_live_odds(odds_date, odds_venue)
    if odds_by_race:
        win_count = place_count = 0
        df_raw['位置賠率'] = np.nan
        for idx, runner in df_raw.iterrows():
            race_no = extract_race_no(runner.get('賽事編號', ''))
            horse_no = normalize_horse_no(runner.get('馬號', ''))
            race_market = odds_by_race.get(race_no, {})
            win_price = race_market.get('WIN', {}).get(horse_no)
            if win_price is not None and win_price > 0:
                df_raw.at[idx, '獨贏賠率'] = win_price
                win_count += 1
            place_price = race_market.get('PLA', {}).get(horse_no)
            if place_price is not None and place_price > 0:
                df_raw.at[idx, '位置賠率'] = place_price
                place_count += 1
        st.session_state['df_data'] = df_raw.copy()
        st.session_state['fetched_prediction_df'] = df_raw.copy()
        
        base_map = {}
        place_base_map = {}
        for race_num, race_markets in odds_by_race.items():
            for horse_no, quote in (race_markets.get('WIN') or {}).items():
                if quote is not None and float(quote) > 1.0:
                    k = f"{normalize_race_id(f'{odds_date.replace('-', '')}-{int(race_num):02d}')}_{normalize_horse_no(horse_no)}"
                    base_map[k] = float(quote)
            for horse_no, quote in (race_markets.get('PLA') or {}).items():
                if quote is not None and float(quote) > 0:
                    k = f"{normalize_race_id(f'{odds_date.replace('-', '')}-{int(race_num):02d}')}_{normalize_horse_no(horse_no)}"
                    place_base_map[k] = float(quote)
        st.session_state['baseline_odds_map'] = base_map
        st.session_state['place_baseline_odds_map'] = place_base_map
        st.session_state['baseline_locked_time'] = f"首個HKJC實報 ({datetime.now(hk_timezone).strftime('%H:%M:%S')})" if base_map else '等候首個HKJC WIN實報'
        st.session_state['odds_server_times_by_race'] = odds_times_by_race or {}
        st.session_state['pending_initial_odds_snapshot'] = {
            'date': odds_date, 'venue': odds_venue,
            'markets': odds_by_race, 'server_times': odds_times_by_race or {},
        }
        auto_odds_status = f'已載入HKJC實際報價：WIN {win_count} 匹、PLACE {place_count} 匹；PLACE缺報價不會以WIN推算'
    else:
        auto_odds_status = f'目前未能取得已開出的官方賠率。'

if auto_odds_status:
    if auto_odds_status.startswith('已自動'):
        st.sidebar.success(auto_odds_status)
    else:
        st.sidebar.warning(auto_odds_status)

if '獨贏賠率' not in df_raw.columns:
    df_raw['獨贏賠率'] = 10.0
else:
    df_raw['獨贏賠率'] = pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0)

if '位置賠率' not in df_raw.columns:
    df_raw['位置賠率'] = np.nan
else:
    df_raw['位置賠率'] = pd.to_numeric(df_raw['位置賠率'], errors='coerce')
if '賽道' not in df_raw.columns: df_raw['賽道'] = 'A'
if '跑法' not in df_raw.columns: df_raw['跑法'] = '居中'

if 'df_data' not in st.session_state or st.session_state.get('prediction_signature') != prediction_signature:
    st.session_state['df_data'] = df_raw.copy()
    st.session_state['prediction_signature'] = prediction_signature
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
    
    # CSV odds may be stale or placeholders; wait for a valid HKJC quote to
    # seed each runner's monitoring baseline.
    st.session_state['baseline_odds_map'] = {}
    st.session_state['place_baseline_odds_map'] = {}
    st.session_state['baseline_locked_time'] = '等候首個HKJC WIN實報'

if '賽道' not in st.session_state['df_data'].columns: st.session_state['df_data']['賽道'] = 'A'
if '跑法' not in st.session_state['df_data'].columns: st.session_state['df_data']['跑法'] = '居中'

st.info("👇 以下是當前載入的賽事資料。你可以點擊表格修改「跑法」與「賠率」，或透過下方的更新面板抓取即時賠率與凍結單場盤口。")

edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '跑法', '賽道', '獨贏賠率', '位置賠率']
for c in edit_columns:
    if c not in st.session_state['df_data'].columns:
        st.session_state['df_data'][c] = ''
df_editable = st.session_state['df_data'][edit_columns].copy()

editor_key = f"odds_editor_{selected_csv_name}_{st.session_state.get('odds_editor_version', 0)}"
edited_df = st.data_editor(
    df_editable,
    disabled=['賽事編號', '馬號', '馬名', '排位檔位', '賽道'],
    column_config={
        "跑法": st.column_config.SelectboxColumn("跑法", options=["放頭", "跟前", "居中", "後上"], required=True)
    },
    use_container_width=True,
    hide_index=True,
    key=editor_key
)

st.session_state['df_data']['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
st.session_state['df_data']['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce')
st.session_state['df_data']['跑法'] = edited_df['跑法']

df = st.session_state['df_data'].copy()

sample_id_for_dl = str(df['賽事編號'].iloc[0]).strip() if len(df) else ""
dl_date_match = re.search(r"(20\d{6})", sample_id_for_dl)
dl_date_tag = dl_date_match.group(1) if dl_date_match else datetime.now(hk_timezone).strftime("%Y%m%d")
snapshot_csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
st.sidebar.download_button(
    f'📸 下載【買時賠率】快照 (prediction_{dl_date_tag}.csv)',
    data=snapshot_csv_bytes,
    file_name=f'prediction_{dl_date_tag}.csv',
    mime='text/csv',
    use_container_width=True,
)

baseline_map = st.session_state.get('baseline_odds_map', {})
df['_horse_uid'] = df.apply(lambda r: f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}", axis=1)
df['基準賠率'] = df['_horse_uid'].map(baseline_map).fillna(df['獨贏賠率'])
df['基準賠率'] = pd.to_numeric(df['基準賠率'], errors='coerce').fillna(df['獨贏賠率'])
df['odds_drop_pct'] = ((df['基準賠率'] - df['獨贏賠率']) / df['基準賠率'].replace(0, np.nan)).fillna(0.0) * 100.0

def get_odds_move_badge(pct):
    if pct >= 20.0: return f"🔥 大落飛 (-{pct:.0f}%)"
    elif pct >= 10.0: return f"📉 落飛 (-{pct:.0f}%)"
    elif pct <= -20.0: return f"🥶 回飛 (+{abs(pct):.0f}%)"
    else: return "➖ 穩定"

df['資金流向'] = df['odds_drop_pct'].map(get_odds_move_badge)

df['market_prob'] = 1 / df['獨贏賠率']
prob_sum = df.groupby('賽事編號')['market_prob'].transform('sum')
df['market_implied_prob'] = df['market_prob'] / prob_sum
df['odds_rank'] = df.groupby('賽事編號')['獨贏賠率'].rank(method='min')
df['is_favorite'] = (df['odds_rank'] == 1).astype(int)
df['排位檔位'] = pd.to_numeric(df['排位檔位'], errors='coerce').fillna(7)

if '實際負磅' in df.columns:
    df['實際負磅'] = pd.to_numeric(df['實際負磅'], errors='coerce').fillna(120)
    avg_weight = df.groupby('賽事編號')['實際負磅'].transform('mean')
    df['weight_diff'] = df['實際負磅'] - avg_weight
    df['weight_rank'] = df.groupby('賽事編號')['實際負磅'].rank(ascending=False, method='min')
else:
    df['weight_diff'] = 0.0
    df['weight_rank'] = 6.0

rank_source = df['名次'] if '名次' in df.columns else pd.Series(99, index=df.index)
df['numeric_rank'] = pd.to_numeric(rank_source, errors='coerce').fillna(99)

if '距離' not in df.columns: df['距離'] = 1200
df['距離'] = pd.to_numeric(df['距離'], errors='coerce').fillna(1200).astype(int)

if '場地' not in df.columns: df['場地'] = '草地'

hist_stats = load_historical_stats()
if hist_stats is not None:
    (
        jockey_stats, trainer_stats, horse_stats, combo_stats,
        horse_last_rank_stats, horse_surface_stats, horse_dist_stats
    ) = hist_stats
    
    df['_clean_jockey'] = df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_trainer'] = df['練馬師'].astype(str).str.replace(' ', '')
    df['_clean_horse'] = df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_surface'] = df['場地'].astype(str).str.strip()
    df['_clean_dist'] = df['距離'].astype(int)
    
    df = df.merge(jockey_stats, on='_clean_jockey', how='left')
    df = df.merge(trainer_stats, on='_clean_trainer', how='left')
    df = df.merge(horse_stats, on='_clean_horse', how='left')
    df = df.merge(combo_stats, on=['_clean_jockey', '_clean_trainer'], how='left')
    df = df.merge(horse_last_rank_stats, on='_clean_horse', how='left')
    df = df.merge(horse_surface_stats, on=['_clean_horse', '_clean_surface'], how='left')
    df = df.merge(horse_dist_stats, on=['_clean_horse', '_clean_dist'], how='left')
    
    df['jockey_win_rate'] = df['hist_jockey_win_rate'].fillna(0.08)
    df['trainer_win_rate'] = df['hist_trainer_win_rate'].fillna(0.08)
    df['recent_jockey_win_rate'] = df.get('recent_jockey_win_rate', df['jockey_win_rate']).fillna(df['jockey_win_rate'])
    df['recent_trainer_win_rate'] = df.get('recent_trainer_win_rate', df['trainer_win_rate']).fillna(df['trainer_win_rate'])
    df['horse_win_rate'] = df['hist_horse_win_rate'].fillna(0.05)
    df['combo_win_rate'] = df['hist_combo_win_rate'].fillna(0.05)
    
    df['real_last_rank'] = df['hist_horse_last_rank'].fillna(6.5)
    df['real_surface_win_rate'] = df['hist_horse_surface_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    df['real_dist_win_rate'] = df['hist_horse_dist_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    
    if use_dynamic_form:
        df['horse_last_rank'] = df['real_last_rank']
        df['horse_surface_win_rate'] = df['real_surface_win_rate']
        df['horse_dist_win_rate'] = df['real_dist_win_rate']
    else:
        df['horse_last_rank'] = 6.0
        df['horse_surface_win_rate'] = 0.08
        df['horse_dist_win_rate'] = 0.08
    
    df.drop(columns=[
        '_clean_jockey', '_clean_trainer', '_clean_horse', '_clean_surface', '_clean_dist',
        'hist_jockey_win_rate', 'hist_trainer_win_rate', 'hist_horse_win_rate', 'hist_combo_win_rate',
        'hist_horse_last_rank', 'hist_horse_surface_win_rate', 'hist_horse_dist_win_rate'
    ], inplace=True, errors='ignore')
else:
    st.sidebar.error("❌ 未能讀取歷史大表，目前正在使用預設值！")
    df['jockey_win_rate'] = df.get('jockey_win_rate', 0.08)
    df['trainer_win_rate'] = df.get('trainer_win_rate', 0.08)
    df['recent_jockey_win_rate'] = df.get('recent_jockey_win_rate', 0.08)
    df['recent_trainer_win_rate'] = df.get('recent_trainer_win_rate', 0.08)
    df['combo_win_rate'] = df.get('combo_win_rate', 0.05)
    df['horse_win_rate'] = df.get('horse_win_rate', 0.05)
    df['real_last_rank'] = 6.5
    df['real_surface_win_rate'] = 0.08
    df['real_dist_win_rate'] = 0.08
    df['horse_last_rank'] = 6.0
    df['horse_surface_win_rate'] = 0.08
    df['horse_dist_win_rate'] = 0.08

feature_cols = FEATURE_COLS_15
for col in feature_cols:
    if col not in df.columns: df[col] = 0.0

X_predict = df[feature_cols].fillna(0)
df['pred_win_prob'] = model_win.predict_proba(X_predict)[:, 1]
if model_place is not None:
    df['pred_place_prob'] = model_place.predict_proba(X_predict)[:, 1]
else:
    df['pred_place_prob'] = df['pred_win_prob']

pending_initial_snapshot = st.session_state.pop('pending_initial_odds_snapshot', None)
if pending_initial_snapshot:
    try:
        initial_model_context = build_snapshot_model_context(
            df, pending_initial_snapshot.get('markets', {})
        )
        st.session_state.pop('audit_model_context_error', None)
    except Exception as exc:
        initial_model_context = {}
        st.session_state['audit_model_context_error'] = f'首次快照模型重評估失敗（官方賠率仍會保存）：{type(exc).__name__}: {exc}'
    for race_num, race_markets in pending_initial_snapshot.get('markets', {}).items():
        try:
            recorded_at, _ = record_single_race_odds(
                pending_initial_snapshot['date'], pending_initial_snapshot['venue'],
                race_num, race_markets, pending_initial_snapshot.get('server_times', {}),
                initial_model_context.get(int(race_num), {}),
            )
            st.session_state['odds_last_checked_by_race'][int(race_num)] = recorded_at
        except Exception as exc:
            st.session_state['audit_local_error'] = f'首次官方賠率快照保存失敗（第{race_num}場）：{type(exc).__name__}: {exc}'

try:
    persist_race_entry_context(df)
except Exception as exc:
    st.session_state['audit_local_error'] = f'賽前資料保存失敗：{type(exc).__name__}: {exc}'

df['win_ev'] = df['pred_win_prob'] * df['獨贏賠率']
df['place_ev'] = df['pred_place_prob'] * df['位置賠率']
df['ev'] = df['win_ev'] 

drop_bonus = np.clip(df['odds_drop_pct'] / 100.0, -0.35, 0.30)
total_tactical_adj = 1.0 + drop_bonus
df['smart_score_win'] = df['pred_win_prob'] * total_tactical_adj
df['smart_score_place'] = df['pred_place_prob'] * total_tactical_adj
df['smart_score'] = df['smart_score_win']

safe_win_odds = np.maximum(df['獨贏賠率'], 1.01)
df['kelly_pct_win'] = np.clip((df['win_ev'] - 1.0) / (safe_win_odds - 1.0), 0.0, 1.0)
df['suggested_stake_win'] = np.floor((total_bankroll * df['kelly_pct_win'] * kelly_fraction) / 10) * 10
df['suggested_stake'] = df['suggested_stake_win']

safe_place_odds = np.maximum(df['位置賠率'], 1.01)
df['kelly_pct_place'] = np.clip((df['place_ev'] - 1.0) / (safe_place_odds - 1.0), 0.0, 1.0)
df['suggested_stake_place'] = np.floor((total_bankroll * df['kelly_pct_place'] * kelly_fraction) / 10) * 10

def filter_and_sort_group(group, mode, min_ev_val, min_odds_val, max_odds_val, bet_type="win"):
    if bet_type == "win":
        cond = (group['win_ev'] >= min_ev_val) & (group['獨贏賠率'] >= min_odds_val) & (group['獨贏賠率'] <= max_odds_val)
        if mode.startswith("🛡️"):
            cond = cond & (group['odds_drop_pct'] > -25.0)
            sort_col = 'smart_score_win'
        elif mode.startswith("🎯"):
            sort_col = 'pred_win_prob'
        else:
            sort_col = 'win_ev'
    else:
        min_pla_odds = max(1.01, min_odds_val / 3.0)
        cond = (group['place_ev'] >= min_ev_val) & (group['位置賠率'] >= min_pla_odds)
        if mode.startswith("🛡️"):
            cond = cond & (group['odds_drop_pct'] > -25.0)
            sort_col = 'smart_score_place'
        elif mode.startswith("🎯"):
            sort_col = 'pred_place_prob'
        else:
            sort_col = 'place_ev'
            
    return group[cond].sort_values(by=sort_col, ascending=False).reset_index(drop=True)

df_backtest = df.copy()
backtest_ready = False
if backtest_file is not None:
    try:
        try:
            result_raw = pd.read_csv(backtest_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            backtest_file.seek(0)
            result_raw = pd.read_csv(backtest_file, encoding='cp950')
        result_source_label = '手動上傳賽果'
        
        required_result_cols = ['賽事編號', '馬號', '名次']
        missing_result_cols = [c for c in required_result_cols if c not in result_raw.columns]
        if missing_result_cols:
            st.sidebar.error(f"回測 CSV 缺少欄位：{missing_result_cols}")
        else:
            result_map = result_raw.copy()
            result_map['_race_key'] = result_map['賽事編號'].map(normalize_race_id)
            result_map['_horse_key'] = result_map['馬號'].map(normalize_horse_no)
            rename_map = {'名次': '_result_rank'}
            if '獨贏賠率' in result_map.columns:
                rename_map['獨贏賠率'] = '_result_win_odds'
            if '位置賠率' in result_map.columns:
                rename_map['位置賠率'] = '_result_place_odds'
            result_map = result_map.drop_duplicates(['_race_key', '_horse_key'], keep='last')
            result_map = result_map[['_race_key', '_horse_key'] + [c for c in rename_map if c in result_map.columns]]
            result_map = result_map.rename(columns=rename_map)

            df_backtest['_race_key'] = df_backtest['賽事編號'].map(normalize_race_id)
            df_backtest['_horse_key'] = df_backtest['馬號'].map(normalize_horse_no)
            df_backtest = df_backtest.merge(
                result_map, on=['_race_key', '_horse_key'], how='left', validate='many_to_one', indicator='_result_match'
            )
            df_backtest['名次'] = df_backtest['_result_rank']
            df_backtest['numeric_rank'] = pd.to_numeric(df_backtest['名次'], errors='coerce').fillna(99)
            if '_result_win_odds' in df_backtest.columns:
                df_backtest['回測獨贏賠率'] = pd.to_numeric(df_backtest['_result_win_odds'], errors='coerce')
            if '_result_place_odds' in df_backtest.columns:
                df_backtest['回測位置賠率'] = pd.to_numeric(df_backtest['_result_place_odds'], errors='coerce')
            has_official_result = df_backtest['_result_match'].eq('both') & df_backtest['_result_rank'].notna()
            matched = int(has_official_result.sum())
            backtest_ready = matched > 0
            if backtest_ready:
                df_backtest = df_backtest[df_backtest['_result_rank'].notna()].copy()
                st.sidebar.success(f"✅ {result_source_label}已配對：{matched} 匹")
            else:
                st.sidebar.warning("未能按賽事編號＋馬號配對到名次。")
    except Exception as exc:
        st.sidebar.error(f"讀取回測 CSV 失敗：{type(exc).__name__}: {exc}")

st.markdown("---")
tab1, tab2 = st.tabs(["🎯 各場次預測推薦與單場凍結", "📈 歷史回測 (買時選馬 vs 最終派彩)"])

with tab1:
    st.subheader("⚡ 臨場賠率更新中心與單場盤口凍結")
    
    col_v, col_d, col_r, col_t = st.columns([1.2, 1.8, 1.2, 1.8])
    fetched_venue = str(df_raw['racecourse_code'].iloc[0]) if 'racecourse_code' in df_raw.columns else ''
    venue_default_index = 1 if fetched_venue == 'ST' else 0
    venue_input = col_v.selectbox("賽事場地", ["HV (跑馬地)", "ST (沙田)"], index=venue_default_index)
    venue_code = "HV" if venue_input.startswith("HV") else "ST"

    sample_id = str(df_raw['賽事編號'].iloc[0]).strip() if len(df_raw) else ""
    date_match = re.search(r"(20\d{6})", sample_id)
    if date_match:
        raw_date = date_match.group(1)
        auto_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    else:
        auto_date = datetime.now().strftime("%Y-%m-%d")
    api_date = col_d.text_input("API 查詢日期 (YYYY-MM-DD)", value=auto_date, key=f"api_date_{prediction_signature}")

    race_series = df_raw['賽事編號'].map(extract_race_no)
    races_available = sorted(int(x) for x in race_series.dropna().unique())
    if races_available:
        target_race = col_r.selectbox("當前操作／查看場次", races_available)
    else:
        target_race = col_r.number_input("當前操作／查看場次", min_value=1, max_value=15, value=1, step=1)

    target_time = col_t.time_input("預定開跑時間", value=datetime.strptime("18:45", "%H:%M").time())
    
    try:
        race_date = datetime.strptime(api_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        st.error("API 日期格式必須是 YYYY-MM-DD。")
        st.stop()

    target_datetime = datetime.combine(race_date, target_time).replace(tzinfo=hk_timezone)
    target_iso = target_datetime.isoformat()

    timer_html = r"""
    <div id="race-countdown" style="font: bold 30px monospace; text-align: center; padding: 8px;">
      正在啟動倒數計時器…
    </div>
    <script>
    const targetEpochMs = Date.parse("__TARGET_ISO__");
    const timerElement = document.getElementById("race-countdown");
    const startWallMs = Date.now();
    const startMonoMs = performance.now();

    function updateCountdown() {
      const elapsedMs = performance.now() - startMonoMs;
      const remainingMs = targetEpochMs - (startWallMs + elapsedMs);

      if (remainingMs > 0) {
        const remainingUs = Math.ceil(remainingMs * 1000);
        const totalSeconds = Math.floor(remainingUs / 1000000);
        const microseconds = remainingUs % 1000000;
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;

        timerElement.style.color = "#1769aa";
        timerElement.textContent =
          `${String(hours).padStart(2, "0")}:` +
          `${String(minutes).padStart(2, "0")}:` +
          `${String(seconds).padStart(2, "0")}.` +
          `${String(microseconds).padStart(6, "0")}`;
        requestAnimationFrame(updateCountdown);
      } else if (remainingMs > -300000) {
        timerElement.style.color = "#b26a00";
        timerElement.textContent = "已到預定開跑時間";
        requestAnimationFrame(updateCountdown);
      } else {
        timerElement.style.color = "#b00020";
        timerElement.textContent = "賽事可能已開跑或結束";
      }
    }
    requestAnimationFrame(updateCountdown);
    </script>
    """.replace("__TARGET_ISO__", target_iso)

    components.html(timer_html, height=70, scrolling=False)

    with st.expander("🔊 HKJC 官方賽日收音機（現場評述）", expanded=False):
        st.caption("選擇官方頁面上的語言並按播放。")
        st.markdown("[在新分頁開啟 HKJC 官方賽日收音機](https://racing.hkjc.com/zh-hk/showcase/live)")
        components.iframe("https://racing.hkjc.com/zh-hk/showcase/live", height=620, scrolling=True)

    auto_col, interval_col, button_col, lock_col = st.columns([1.5, 1.3, 2.0, 2.0])
    monitor_selected_only = auto_col.checkbox("只監測所選場次", value=True, help="建議開啟：只查目前選定的場次，減少請求量。")
    auto_refresh = auto_col.checkbox("自動監測", value=False)
    refresh_seconds = interval_col.selectbox(
        "更新間隔（秒）", [10, 15, 30, 60, 120, 300], index=0, disabled=not auto_refresh
    )
    manual_refresh = button_col.button(f"🔄 立即更新第 {target_race} 場賠率" if monitor_selected_only else "🔄 立即更新未凍結場次", use_container_width=True)
    lock_baseline = lock_col.button("📌 將最近官方快照鎖定為落飛基準", use_container_width=True, help="先按「立即更新」取得真實HKJC報價，再按此鍵鎖定目前已記錄的官方WIN／PLACE報價；不會使用CSV佔位值。")
    st.caption("每匹馬首次有效官方報價會先作為監測基準；之後可鎖定最近快照重新計算落飛。輪詢間隔是程式查詢頻率，不代表HKJC每隔同樣時間必定有新報價。自動監測由開啟的Streamlit工作階段觸發；頁面關閉／服務休眠時不會持續輪詢。")
    auto_result_check = st.checkbox(
        "自動檢查並配對已公布賽果",
        value=True,
        help="每次監測更新時，依次檢查官方完賽結果；每次至多查一場，結果未公布時稍後再查。若要完整累積全部場次，請關閉「只監測所選場次」。"
    )

    f_col1, f_col2, f_col3 = st.columns([2.2, 1.8, 2.6])
    is_target_frozen = int(target_race) in st.session_state['frozen_races']
    if not is_target_frozen:
        if f_col1.button(f"🔒 凍結第 {target_race} 場【買時賠率】 (下注後點擊)", use_container_width=True, type="primary"):
            st.session_state['frozen_races'][int(target_race)] = datetime.now(hk_timezone).strftime("%H:%M:%S")
            st.rerun()
    else:
        frozen_t = st.session_state['frozen_races'][int(target_race)]
        if f_col1.button(f"🔓 解除第 {target_race} 場凍結 (已鎖於 {frozen_t})", use_container_width=True):
            st.session_state['frozen_races'].pop(int(target_race), None)
            st.rerun()

    if f_col2.button("🔓 全部場次解除凍結", use_container_width=True, disabled=len(st.session_state['frozen_races']) == 0):
        st.session_state['frozen_races'] = {}
        st.rerun()

    frozen_list_str = ", ".join([f"R{r}({t})" for r, t in sorted(st.session_state['frozen_races'].items())]) if st.session_state['frozen_races'] else "尚無凍結場次"
    f_col3.info(f"🔒 已凍結買時盤口：{frozen_list_str}")

    if lock_baseline:
        base_map = {}
        place_base_map = {}
        baseline_rows = pd.DataFrame(st.session_state.get('odds_monitor_history', []))
        if not baseline_rows.empty:
            baseline_rows = baseline_rows[
                (baseline_rows['賽日'].astype(str) == api_date)
                & (baseline_rows['場地'].astype(str) == venue_code)
            ].copy()
            baseline_rows['_capture_dt'] = pd.to_datetime(
                baseline_rows['查詢時間(HKT)'], errors='coerce', utc=True
            )
            baseline_rows = baseline_rows.sort_values('_capture_dt').drop_duplicates(
                ['場次', '馬號'], keep='last'
            )
            for _, snapshot_row in baseline_rows.iterrows():
                baseline_race_id = normalize_race_id(
                    api_date.replace('-', '') + f"-{int(snapshot_row['場次']):02d}"
                )
                k = f"{baseline_race_id}_{normalize_horse_no(snapshot_row['馬號'])}"
                win_value = pd.to_numeric(pd.Series([snapshot_row.get('WIN賠率')]), errors='coerce').iloc[0]
                place_value = pd.to_numeric(pd.Series([snapshot_row.get('PLACE賠率')]), errors='coerce').iloc[0]
                if pd.notna(win_value) and float(win_value) > 1.0:
                    base_map[k] = float(win_value)
                if pd.notna(place_value) and float(place_value) > 0:
                    place_base_map[k] = float(place_value)
        if base_map:
            st.session_state['baseline_odds_map'] = base_map
            st.session_state['place_baseline_odds_map'] = place_base_map
            st.session_state['baseline_locked_time'] = datetime.now(hk_timezone).strftime("%Y-%m-%d %H:%M:%S%z")
            st.rerun()
        else:
            st.warning('尚無此日的有效HKJC WIN實報快照；請先按「立即更新」，不要以預測CSV或10.0佔位值作基準。')

    auto_refresh_tick = False
    if auto_refresh:
        refresh_tick = st_autorefresh(interval=int(refresh_seconds * 1000), key="hkjc_live_odds_autorefresh")
        previous_tick = st.session_state.get('odds_last_processed_refresh_tick')
        if refresh_tick != previous_tick:
            auto_refresh_tick = True
            st.session_state['odds_last_processed_refresh_tick'] = refresh_tick

    if (auto_refresh_tick or manual_refresh) and not (monitor_selected_only and int(target_race) in st.session_state['frozen_races']):
        if get_supabase_config()[0]:
            try:
                sync_pending_snapshots()
            except Exception as exc:
                st.session_state['audit_local_error'] = f'雲端待同步佇列處理失敗：{type(exc).__name__}: {exc}'
        query_race = int(target_race) if monitor_selected_only else None
        query_started_at = datetime.now(hk_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
        if query_race is not None:
            st.session_state['odds_last_checked_by_race'][int(query_race)] = query_started_at
        with st.spinner(f"正在讀取第 {target_race} 場 WIN／PLACE 實際報價…" if query_race else "正在讀取未凍結場次 WIN／PLACE 實際報價…"):
            live_by_race, server_times_by_race, error = fetch_live_odds(api_date, venue_code, query_race)

        if live_by_race:
            snapshot_live_by_race = {
                int(race_no): markets for race_no, markets in live_by_race.items()
                if int(race_no) not in st.session_state['frozen_races']
            }
            try:
                snapshot_model_context = build_snapshot_model_context(df, snapshot_live_by_race)
                st.session_state.pop('audit_model_context_error', None)
            except Exception as exc:
                snapshot_model_context = {}
                st.session_state['audit_model_context_error'] = f'本輪快照模型重評估失敗（官方賠率仍會保存）：{type(exc).__name__}: {exc}'
            df_temp = st.session_state['df_data'].copy()
            row_races = df_temp['賽事編號'].map(extract_race_no)
            row_horses = df_temp['馬號'].map(normalize_horse_no)
            updated_races_count = 0
            skipped_frozen = []
            updated_win_count = 0
            updated_place_count = 0

            for race_no, markets in live_by_race.items():
                r_int = int(race_no)
                if r_int in st.session_state['frozen_races']:
                    skipped_frozen.append(f"R{r_int}")
                    continue
                updated_races_count += 1
                race_mask = row_races == r_int
                # 不把舊的CSV／舊session PLACE值冒充成這次HKJC即時報價。
                df_temp.loc[race_mask, '位置賠率'] = np.nan
                for horse_no, value in markets['WIN'].items():
                    horse_mask = race_mask & (row_horses == horse_no)
                    df_temp.loc[horse_mask, '獨贏賠率'] = value
                    updated_win_count += int(horse_mask.sum())
                for horse_no, value in markets['PLA'].items():
                    horse_mask = race_mask & (row_horses == horse_no)
                    df_temp.loc[horse_mask, '位置賠率'] = value
                    updated_place_count += int(horse_mask.sum())

            if st.session_state.get('baseline_locked_time') in ("初始載入值", "等候首個HKJC WIN實報"):
                base_map = {}
                place_base_map = {}
                for race_num, race_markets in live_by_race.items():
                    race_id_base = normalize_race_id(f"{api_date.replace('-', '')}-{int(race_num):02d}")
                    for horse_no, quote in (race_markets.get('WIN') or {}).items():
                        if quote is not None and float(quote) > 1.0:
                            base_map[f"{race_id_base}_{normalize_horse_no(horse_no)}"] = float(quote)
                    for horse_no, quote in (race_markets.get('PLA') or {}).items():
                        if quote is not None and float(quote) > 0:
                            place_base_map[f"{race_id_base}_{normalize_horse_no(horse_no)}"] = float(quote)
                st.session_state['baseline_odds_map'] = base_map
                st.session_state['place_baseline_odds_map'] = place_base_map
                st.session_state['baseline_locked_time'] = f"首個HKJC實報 ({datetime.now(hk_timezone).strftime('%H:%M:%S')})" if base_map else '等候首個HKJC WIN實報'

            st.session_state['df_data'] = df_temp
            st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
            skip_msg = f"（已保護跳過凍結場次：{', '.join(skipped_frozen)}）" if skipped_frozen else ""
            query_time = datetime.now(hk_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            st.session_state['odds_server_times_by_race'].update(server_times_by_race or {})
            for race_num, race_markets in snapshot_live_by_race.items():
                recorded_at, _ = record_single_race_odds(
                    api_date, venue_code, race_num, race_markets, server_times_by_race,
                    snapshot_model_context.get(int(race_num), {}),
                )
                st.session_state['odds_last_checked_by_race'][int(race_num)] = recorded_at
            focused_times = (server_times_by_race or {}).get(int(target_race), {}) if monitor_selected_only else {}
            source_times = f"HKJC WIN更新：{focused_times.get('WIN') or '未提供'}；PLACE更新：{focused_times.get('PLA') or '未提供'}" if monitor_selected_only else "已保存各場HKJC pool更新時間"
            st.session_state['odds_last_status'] = f"✅ 查詢完成：{updated_races_count} 場；WIN實報 {updated_win_count} 匹、PLACE實報 {updated_place_count} 匹；本機查詢 {query_time}；{source_times}。{skip_msg}"
            st.session_state['odds_last_error'] = None
            if auto_result_check:
                race_numbers = df['賽事編號'].map(extract_race_no).dropna().astype(int).unique().tolist()
                auto_check_race_results(api_date, venue_code, race_numbers, prefer_race=target_race)
            st.rerun()
        else:
            st.session_state['odds_last_error'] = f"⚠️ 無法取得賠率：{error}"
            st.session_state['odds_last_status'] = None
            if auto_result_check and (auto_refresh_tick or manual_refresh):
                race_numbers = df['賽事編號'].map(extract_race_no).dropna().astype(int).unique().tolist()
                auto_check_race_results(api_date, venue_code, race_numbers, prefer_race=target_race)
            if manual_refresh: st.rerun()
    elif monitor_selected_only and int(target_race) in st.session_state['frozen_races'] and (auto_refresh_tick or manual_refresh):
        st.session_state['odds_last_error'] = f"第 {target_race} 場已凍結，為保留買時盤口已停止該場監測。"
        if auto_result_check:
            race_numbers = df['賽事編號'].map(extract_race_no).dropna().astype(int).unique().tolist()
            auto_check_race_results(api_date, venue_code, race_numbers, prefer_race=target_race)

    locked_time_str = st.session_state.get('baseline_locked_time', '未鎖定')
    form_mode_str = "🔥 真實近況與同程算分" if use_dynamic_form else "🎯 1004 爆冷模式 (固定常數算分)"
    engine_str = "雙渦輪 (Win + Place 獨立模型)" if model_place is not None else "單引擎 (僅獨贏模型)"
    st.caption(f"📌 早盤基準：`{locked_time_str}` ｜ 排序：`{strategy_mode}` ｜ 算分引擎：`{form_mode_str}` ｜ 模型：`{engine_str}`")

    if st.session_state.get('odds_last_error'):
        st.warning(st.session_state['odds_last_error'])
    elif st.session_state.get('odds_last_status'):
        st.success(st.session_state['odds_last_status'])

    st.markdown("---")
    st.subheader("🎯 各場次 AI 雙渦輪智慧投注推薦清單")

    recommendations = []
    for race_id, group in df.groupby('賽事編號'):
        r_no = extract_race_no(race_id)
        freeze_badge = f"🔒 已凍結 ({st.session_state['frozen_races'][r_no]})" if r_no in st.session_state['frozen_races'] else "🟢 即時跳動中"
        
        sorted_win = filter_and_sort_group(group, strategy_mode, min_win_ev, min_odds, max_odds, bet_type="win")
        sorted_place = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")

        win_pick = "⚠️ 無達標獨贏馬"
        hot_str = "-"
        if len(sorted_win) >= 1:
            top1 = sorted_win.iloc[0]
            last_r_str = f"上仗:{int(top1['real_last_rank'])}" if top1['real_last_rank'] != 6.5 else "上仗:新馬/無"
            j_hot = "🔥" if (top1['recent_jockey_win_rate'] - top1['jockey_win_rate']) >= 0.04 else ""
            t_hot = "🔥" if (top1['recent_trainer_win_rate'] - top1['trainer_win_rate']) >= 0.04 else ""
            hot_str = f"{top1['騎師']}{j_hot} / {top1['練馬師']}{t_hot}"
            win_pick = (
                f"馬號 {top1['馬號']} ({top1['馬名']}) "
                f"[AI勝率:{top1['pred_win_prob']*100:.1f}% | {last_r_str} | 賠率:{top1['基準賠率']:.1f}➔{top1['獨贏賠率']:.1f} | 獨EV:{top1['win_ev']:.2f}] "
                f"💰${top1['suggested_stake_win']:.0f} {top1['資金流向']}"
            )
            
        if len(sorted_place) >= 2:
            p1 = sorted_place.iloc[0]
            p2 = sorted_place.iloc[1]
            q_pick = f"{p1['馬號']} + {p2['馬號']} ({p1['馬名']} / {p2['馬名']}) [入位率:{p1['pred_place_prob']*100:.0f}% & {p2['pred_place_prob']*100:.0f}%]"
            if len(sorted_place) >= 3:
                p3 = sorted_place.iloc[2]
                qp_pick = f"{p1['馬號']} + {p2['馬號']} 或 {p1['馬號']} + {p3['馬號']}"
            else:
                qp_pick = f"{p1['馬號']} + {p2['馬號']} (僅一注 QP)"
        elif len(sorted_place) == 1:
            p1 = sorted_place.iloc[0]
            q_pick = f"單膽位置: {p1['馬號']} ({p1['馬名']}) [位EV:{p1['place_ev']:.2f}] 💰${p1['suggested_stake_place']:.0f}"
            qp_pick = "⚠️ 僅一匹位置達標 (不推薦 QP)"
        else:
            q_pick = "⚠️ 無達標位置馬"
            qp_pick = "⚠️ 無達標位置馬"

        if len(sorted_win) >= 1 or len(sorted_place) >= 1:
            ref_row = sorted_win.iloc[0] if len(sorted_win) >= 1 else sorted_place.iloc[0]
            if hot_str == "-":
                j_hot = "🔥" if (ref_row['recent_jockey_win_rate'] - ref_row['jockey_win_rate']) >= 0.04 else ""
                t_hot = "🔥" if (ref_row['recent_trainer_win_rate'] - ref_row['trainer_win_rate']) >= 0.04 else ""
                hot_str = f"{ref_row['騎師']}{j_hot} / {ref_row['練馬師']}{t_hot}"
                
            recommendations.append({
                '賽事編號': race_id,
                '賽道': ref_row.get('賽道', 'A'),
                '盤口狀態': freeze_badge,
                '🎯 獨贏首選 (Win Engine)': win_pick,
                '近況手風': hot_str,
                '🥉 位置/連贏推薦 (Place Engine)': q_pick,
                '🔗 位置Q推薦 (Place Engine)': qp_pick
            })

    rec_df = pd.DataFrame(recommendations)
    if rec_df.empty:
        st.warning("⚠️ 沒有符合當前篩選條件的馬匹。")
    else:
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

    status_tag = f"🔒 已凍結於 {st.session_state['frozen_races'][int(target_race)]}" if is_target_frozen else "🟢 盤口即時更新中"
    race_detail_df = df[df['賽事編號'].map(extract_race_no) == target_race].copy()
    st.markdown(f"#### 🔍 第 {target_race} 場 — 臨場落飛與雙模型實力深度雷達 ({status_tag})")
    
    if not race_detail_df.empty:
        missing_place_count = int(pd.to_numeric(race_detail_df['位置賠率'], errors='coerce').isna().sum())
        if missing_place_count:
            st.caption(f"提示：本場有 {missing_place_count} 匹馬尚無有效PLACE實報；系統不會由WIN賠率推算，該等馬匹不會產生有效位置EV建議。")
        if strategy_mode.startswith("🛡️"):
            sort_col_detail = 'smart_score_win'
        elif strategy_mode.startswith("🎯"):
            sort_col_detail = 'pred_win_prob'
        else:
            sort_col_detail = 'win_ev'
            
        race_detail_df = race_detail_df.sort_values(by=sort_col_detail, ascending=False).reset_index(drop=True)
        display_detail = pd.DataFrame({
            '馬號': race_detail_df['馬號'],
            '馬名': race_detail_df['馬名'],
            '騎師(近50場)': race_detail_df['騎師'] + " (" + (race_detail_df['recent_jockey_win_rate'] * 100).round(1).astype(str) + "%)",
            '練馬師': race_detail_df['練馬師'],
            '檔位': race_detail_df['排位檔位'],
            '跑法': race_detail_df['跑法'],
            '上仗名次': race_detail_df['real_last_rank'].map(lambda x: "新馬/無" if x == 6.5 else f"第 {int(x)} 名"),
            '同程勝率': (race_detail_df['real_dist_win_rate'] * 100).round(1).astype(str) + '%',
            'AI 勝率': (race_detail_df['pred_win_prob'] * 100).round(1).astype(str) + '%',
            'AI 入位率': (race_detail_df['pred_place_prob'] * 100).round(1).astype(str) + '%',
            '基準獨贏': race_detail_df['基準賠率'].round(1),
            '最新獨贏': race_detail_df['獨贏賠率'].round(1),
            '位置賠率': race_detail_df['位置賠率'].round(1),
            '資金流向': race_detail_df['資金流向'],
            '獨贏 EV': race_detail_df['win_ev'].round(2),
            '位置 EV': race_detail_df['place_ev'].round(2),
            '抗噪分': (race_detail_df['smart_score_win'] * 100).round(1),
            '獨贏注碼': race_detail_df['suggested_stake_win'].apply(lambda x: f"${x:.0f}"),
            '位置注碼': race_detail_df['suggested_stake_place'].apply(lambda x: f"${x:.0f}")
        })
        st.dataframe(display_detail, use_container_width=True, hide_index=True)

    st.markdown(f"#### 📉 第 {target_race} 場落飛監測記錄")
    st.caption("每筆快照記錄程式查詢時間（香港時間至微秒）、HKJC WIN／PLACE pool更新時間、官方實報、落飛基準與當輪模型概率／15項輸入／模型簽章；先寫SQLite，再於配置Supabase時同步雲端。未配置雲端備份時，Streamlit Cloud本機SQLite可能隨重啟／重新部署而消失。")
    focus_times = st.session_state.get('odds_server_times_by_race', {}).get(int(target_race), {})
    checked_at = st.session_state.get('odds_last_checked_by_race', {}).get(int(target_race), '尚未查詢')
    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("程式最近查詢時間", checked_at)
    tcol2.metric("HKJC WIN pool更新", str(focus_times.get('WIN') or '尚無回傳'))
    tcol3.metric("HKJC PLACE pool更新", str(focus_times.get('PLA') or '尚無回傳'))
    audit_race_id = normalize_race_id(f"{api_date.replace('-', '')}-{int(target_race):02d}")
    audit_loaded = st.session_state.setdefault('audit_loaded_races', set())
    if audit_race_id not in audit_loaded:
        saved_snapshots, _ = load_race_audit_data(audit_race_id)
        if not saved_snapshots.empty:
            rename_saved = {
                'captured_at': '查詢時間(HKT)', 'race_date': '賽日', 'venue': '場地',
                'race_no': '場次', 'horse_no': '馬號', 'horse_name': '馬名',
                'win_odds': 'WIN賠率', 'place_odds': 'PLACE賠率',
                'win_baseline': 'WIN基準賠率', 'place_baseline': 'PLACE基準賠率',
                'win_drop': 'WIN落飛百分比', 'place_drop': 'PLACE落飛百分比',
                'win_pool_updated_at': 'HKJC_WIN_LastUpdate',
                'place_pool_updated_at': 'HKJC_PLACE_LastUpdate',
            }
            saved_snapshots = saved_snapshots.rename(columns=rename_saved)
            current_saved = st.session_state.setdefault('odds_monitor_history', [])
            known = {(str(item.get('馬號')), str(item.get('查詢時間(HKT)'))) for item in current_saved}
            for saved in saved_snapshots.to_dict(orient='records'):
                key = (str(saved.get('馬號')), str(saved.get('查詢時間(HKT)')))
                if key not in known:
                    current_saved.append(saved)
                    known.add(key)
        audit_loaded.add(audit_race_id)
    monitor_history = pd.DataFrame(st.session_state.get('odds_monitor_history', []))
    if not monitor_history.empty:
        monitor_history = monitor_history[
            (monitor_history['賽日'].astype(str) == api_date)
            & (monitor_history['場地'].astype(str) == venue_code)
            & (pd.to_numeric(monitor_history['場次'], errors='coerce') == int(target_race))
        ].copy()
    if not monitor_history.empty:
        monitor_history = monitor_history.sort_values('查詢時間(HKT)', ascending=False)
        chart_history = monitor_history.copy()
        for chart_col in ['WIN賠率', 'PLACE賠率', 'WIN落飛百分比', 'PLACE落飛百分比']:
            if chart_col not in chart_history.columns:
                chart_history[chart_col] = np.nan
            chart_history[chart_col] = pd.to_numeric(chart_history[chart_col], errors='coerce')
        chart_history['_time'] = pd.to_datetime(chart_history['查詢時間(HKT)'], errors='coerce', utc=True).dt.tz_convert('Asia/Hong_Kong').dt.tz_localize(None)
        chart_history = chart_history.dropna(subset=['_time'])
        chart_history['_horse_label'] = chart_history['馬號'].astype(str) + ' ' + chart_history['馬名'].fillna('').astype(str)
        if not chart_history.empty:
            latest_per_horse = chart_history.sort_values('_time').groupby('_horse_label', as_index=False).tail(1)
            latest_per_horse['_drop_sort'] = pd.to_numeric(latest_per_horse['WIN落飛百分比'], errors='coerce').fillna(-999)
            default_horses = latest_per_horse.sort_values('_drop_sort', ascending=False)['_horse_label'].head(4).tolist()
            horse_options = sorted(chart_history['_horse_label'].dropna().unique().tolist())
            selected_horses = st.multiselect(
                '圖表顯示馬匹（預設顯示最新落飛較多的4匹）',
                options=horse_options,
                default=[h for h in default_horses if h in horse_options],
                key=f'odds_chart_horses_{api_date}_{venue_code}_{int(target_race)}',
            )
            plot_history = chart_history[chart_history['_horse_label'].isin(selected_horses)].sort_values('_time')
            chart_left, chart_right = st.columns(2)
            win_plot = plot_history.pivot_table(index='_time', columns='_horse_label', values='WIN賠率', aggfunc='last').sort_index()
            place_plot = plot_history.pivot_table(index='_time', columns='_horse_label', values='PLACE賠率', aggfunc='last').sort_index()
            drop_plot = plot_history.pivot_table(index='_time', columns='_horse_label', values='WIN落飛百分比', aggfunc='last').sort_index()
            place_drop_plot = plot_history.pivot_table(index='_time', columns='_horse_label', values='PLACE落飛百分比', aggfunc='last').sort_index()
            model_place_plot = plot_history.pivot_table(index='_time', columns='_horse_label', values='snapshot_pred_place_prob', aggfunc='last').sort_index() if 'snapshot_pred_place_prob' in plot_history.columns else pd.DataFrame()
            chart_left.markdown('**WIN賠率走勢（賠率下降代表落飛）**')
            if not win_plot.empty:
                chart_left.line_chart(win_plot, height=260)
            else:
                chart_left.caption('目前沒有可畫的WIN賠率快照。')
            chart_right.markdown('**PLACE賠率走勢（只繪HKJC實報）**')
            if not place_plot.empty:
                chart_right.line_chart(place_plot, height=260)
            else:
                chart_right.caption('目前尚無HKJC PLACE實報快照。')
            st.markdown('**相對基準WIN落飛幅度（%）**')
            if not drop_plot.empty:
                st.line_chart(drop_plot, height=240)
            else:
                st.caption('尚未設定有效WIN基準價，因此暫無落飛百分比。')
            st.markdown('**相對基準PLACE落飛幅度（%）**')
            if not place_drop_plot.empty:
                st.line_chart(place_drop_plot, height=240)
            else:
                st.caption('尚未收到有效HKJC PLACE基準價，因此暫無PLACE落飛百分比。')
            st.markdown('**同一輪官方WIN盤重算的模型入位概率（%）**')
            if not model_place_plot.empty:
                st.line_chart(model_place_plot * 100.0, height=240)
                st.caption('只在該場WIN盤每匹排位馬均有有效HKJC實報時重算；請查看CSV中的snapshot_model_form及模型簽章。')
            else:
                st.caption('尚無完整WIN盤模型重算紀錄；新版本部署後開始收集。')
        st.dataframe(monitor_history.head(300), use_container_width=True, hide_index=True)
        st.download_button(
            f"⬇️ 下載第 {target_race} 場落飛監測CSV",
            data=monitor_history.sort_values('查詢時間(HKT)').to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f"odds_monitor_{api_date.replace('-', '')}_{venue_code}_R{int(target_race):02d}.csv",
            mime='text/csv',
            key=f"download_odds_monitor_{api_date}_{venue_code}_{int(target_race)}",
        )
    else:
        st.caption("尚無監測快照。請啟動自動監測，或按「立即更新」抓取第一筆。")

    st.markdown(f"#### 🏁 第 {target_race} 場 — 頭三名賽後落飛報告")
    target_race_id = normalize_race_id(f"{api_date.replace('-', '')}-{int(target_race):02d}")
    report_df_saved, report_completed_at, report_source_url = load_postrace_report(target_race_id)
    if st.button(f"🔎 現在檢查第 {target_race} 場官方賽果並產生報告", key=f"manual_postrace_{target_race_id}"):
        with st.spinner('正在讀取HKJC官方賽果並配對賽前監測紀錄…'):
            official_results, official_url = fetch_official_race_results(api_date, venue_code, target_race)
            if official_results.empty:
                st.warning('HKJC官方結果尚未公布，或頁面暫時無法讀取；稍後可再按一次檢查。')
            else:
                report_df_saved, _ = make_postrace_report(api_date, venue_code, target_race, official_results, official_url)
                report_completed_at = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S%z')
                report_source_url = official_url
                st.rerun()
    if not report_df_saved.empty:
        st.caption(f"官方賽果配對時間：{report_completed_at} ｜ 資料來源：HKJC LocalResults ｜ 報告只呈現可觀察關聯，不判定因果。")
        st.dataframe(report_df_saved, use_container_width=True, hide_index=True)
        st.download_button(
            f"⬇️ 下載第 {target_race} 場頭三名賽後報告 CSV",
            data=report_df_saved.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f"postrace_top3_{api_date.replace('-', '')}_{venue_code}_R{int(target_race):02d}.csv",
            mime='text/csv',
            key=f"download_postrace_{target_race_id}",
        )
    else:
        st.info('官方賽果公布後會自動配對；也可按上方按鈕立即檢查。報告包含前三名落飛幅度、監測快照數、賽前模型／排位資料及官方結果頁可取得的沿途走位或分段欄位。')
    cloud_url, cloud_key = get_supabase_config()
    if cloud_url and cloud_key:
        st.success('長期雲端備份：已設定Supabase連線。')
    else:
        st.warning('尚未設定Supabase：目前只保存於應用程式本機SQLite；Streamlit Cloud重啟／重新部署時本機檔案可能消失。請依附上的SQL建立雲端表，並在Streamlit Secrets設定SUPABASE_URL及SUPABASE_SERVICE_ROLE_KEY。')
    if st.session_state.get('audit_local_error'):
        st.error(f"SQLite保存錯誤：{st.session_state['audit_local_error']}")
    if st.session_state.get('audit_cloud_error'):
        st.error(f"Supabase同步錯誤：{st.session_state['audit_cloud_error']}")
    if st.session_state.get('audit_model_context_error'):
        st.warning(st.session_state['audit_model_context_error'])

    # === 🎲 WIN + PLACE 模型名次模擬（僅為模型估算，非官方派彩） ===
    st.markdown(f"#### 🎲 第 {target_race} 場 — WIN + PLACE 模型組合模擬")
    st.caption("模擬把獨贏勝率與位置模型入位傾向轉成相對排序強度，再以 Plackett–Luce/Gumbel 方法抽樣名次。它不是HKJC實際派彩預測，模擬次數增加只減少抽樣誤差，不會自動提高模型準確率。")
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    place_model_weight = mc_col1.slider(
        "位置模型權重", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0=只用獨贏模型排序訊號；1=只用位置模型相對入位訊號；中間值為混合。應以較早賽期選權重，再用未參與選擇的後期賽事驗證。",
        key="mc_place_weight"
    )
    mc_num_sims = mc_col2.selectbox("模擬次數", [10000, 50000, 100000], index=1, key="mc_num_sims")
    mc_seed = mc_col3.number_input("固定隨機種子", min_value=0, max_value=2147483647, value=20261025, step=1, key="mc_seed")
    mc_button_disabled = model_place is None or len(race_detail_df) < 3
    if model_place is None:
        st.warning("目前沒有獨立 PLACE 模型；此功能需要 app.py 同目錄的 `my_hkjc_place_model.pkl`，為免把WIN模型假扮成PLACE訊號，已停用模擬。")
    elif len(race_detail_df) < 3:
        st.warning("本場有效馬匹少於3匹，無法模擬頭三名組合。")
    if st.button("🚀 啟動 WIN + PLACE 模型模擬", use_container_width=True, disabled=mc_button_disabled):
        with st.spinner(f"正在模擬 {mc_num_sims:,} 次名次排序…"):
            try:
                df_q, df_qp, df_trio = run_monte_carlo_exotics(
                    race_detail_df,
                    num_sims=int(mc_num_sims),
                    place_weight=float(place_model_weight),
                    random_seed=int(mc_seed),
                )
                st.session_state['mc_results'] = (df_q, df_qp, df_trio)
                st.session_state['mc_target_race'] = target_race
                st.session_state['mc_params'] = {
                    'num_sims': int(mc_num_sims),
                    'place_weight': float(place_model_weight),
                    'random_seed': int(mc_seed),
                }
                st.session_state['mc_results_version'] = 2
            except Exception as exc:
                st.error(f"蒙地卡羅模擬失敗：{type(exc).__name__}: {exc}")

    if (st.session_state.get('mc_results_version') == 2
            and st.session_state.get('mc_results')
            and st.session_state.get('mc_target_race') == target_race):
        df_q, df_qp, df_trio = st.session_state['mc_results']
        mc_saved_params = st.session_state.get('mc_params', {})
        mc_sims_display = mc_saved_params.get('num_sims')
        st.caption(
            f"模型估算；模擬 {mc_sims_display:,} 次｜"
            f"PLACE權重 {mc_saved_params.get('place_weight', '—')}｜"
            f"隨機種子 {mc_saved_params.get('random_seed', '—')}。重跑同一場與相同輸入/參數會得到相同結果。"
        )
        
        if df_q is not None:
            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                st.success("🔗 連贏 (Q) 模型估算")
                st.dataframe(df_q[['H1', 'H2', '組合名稱', 'hits', '模型估算機率', '模型估算公平賠率']].rename(columns={'hits': '模擬命中次數', 'H1': '馬1', 'H2': '馬2'}).style.format({'模型估算機率': '{:.3%}', '模型估算公平賠率': '{:.1f}'}), hide_index=True)
                
            with m_col2:
                st.warning("🥉 位置Q (QP) 模型估算")
                st.dataframe(df_qp[['H1', 'H2', '組合名稱', 'hits', '模型估算機率', '模型估算公平賠率']].rename(columns={'hits': '模擬命中次數', 'H1': '馬1', 'H2': '馬2'}).style.format({'模型估算機率': '{:.3%}', '模型估算公平賠率': '{:.1f}'}), hide_index=True)
                
            with m_col3:
                st.info("🏆 單T (Trio) 模型估算")
                st.dataframe(df_trio[['H1', 'H2', 'H3', '組合名稱', 'hits', '模型估算機率', '模型估算公平賠率']].rename(columns={'hits': '模擬命中次數', 'H1': '馬1', 'H2': '馬2', 'H3': '馬3'}).style.format({'模型估算機率': '{:.3%}', '模型估算公平賠率': '{:.1f}'}), hide_index=True)


    st.markdown("---")
    st.subheader("🛒 虛擬模擬投注站 (Paper Trading)")
    st.caption("開跑前 2~3 分鐘看準上方推薦後，在此鎖定當下買入賠率（下注同時會自動🔒凍結該場盤口）。")

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([1, 2, 1.5, 1.5, 1.5])
    
    sim_race = col_s1.selectbox("場次", races_available if races_available else [1], key="sim_race")
    
    horses_in_race = df[df['賽事編號'].map(extract_race_no) == sim_race].copy()
    horses_in_race['display_name'] = horses_in_race['馬號'].astype(str) + " (" + horses_in_race['馬名'] + ")"
    sim_horse = col_s2.selectbox("選擇馬匹", horses_in_race['display_name'].tolist() if not horses_in_race.empty else ["無資料"], key="sim_horse")
    
    sim_type = col_s3.selectbox("玩法", ["獨贏 (WIN)", "位置 (PLACE)"], key="sim_type")
    sim_amount = col_s4.number_input("注碼", min_value=10, value=100, step=10, key="sim_amount")
    
    if col_s5.button("➕ 確定下注並凍結該場", use_container_width=True, type="primary") and not horses_in_race.empty:
        horse_no_raw = sim_horse.split(" ")[0]
        horse_name_raw = sim_horse.split("(")[1].replace(")", "")
        
        target_row = horses_in_race[horses_in_race['馬號'].astype(str) == horse_no_raw]
        if "WIN" in sim_type:
            current_odds = target_row['獨贏賠率'].values[0]
            bet_type = "WIN"
        else:
            current_odds = target_row['位置賠率'].values[0]
            bet_type = "PLA"
            
        now_str = datetime.now(hk_timezone).strftime("%H:%M:%S")
        new_bet = pd.DataFrame([{
            '下注時間': now_str,
            '賽事編號': target_row['賽事編號'].values[0],
            '場次': sim_race,
            '馬號': horse_no_raw,
            '馬名': horse_name_raw,
            '玩法': bet_type,
            '買入賠率': current_odds,
            '注碼': sim_amount
        }])
        
        st.session_state['simulated_bets'] = pd.concat([st.session_state['simulated_bets'], new_bet], ignore_index=True)
        st.session_state['frozen_races'][int(sim_race)] = now_str
        st.success(f"✅ 成功記錄注單並自動凍結第 {sim_race} 場買時盤口：{horse_no_raw} 號 ({bet_type}) @ 賠率 {current_odds}")

    if not st.session_state['simulated_bets'].empty:
        st.markdown("##### 🧾 目前模擬注單紀錄")
        st.dataframe(st.session_state['simulated_bets'], use_container_width=True, hide_index=True)
        
        dl_col, clear_col = st.columns([2, 2])
        csv_sim = st.session_state['simulated_bets'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        dl_col.download_button(
            label="⬇ 下載模擬注單 (CSV)",
            data=csv_sim,
            file_name=f"simulated_bets_{api_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
        if clear_col.button("🗑 清除所有模擬紀錄", use_container_width=True):
            st.session_state['simulated_bets'] = pd.DataFrame(columns=[
                '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
            ])
            st.rerun()

with tab2:
    st.subheader("📊 雙引擎多彩種策略回測總覽 (真實凱利注碼 vs 官方最終派彩)")
    st.caption(f"目前回測套用之排序邏輯：`{strategy_mode}` ｜ 算分引擎：`{form_mode_str}` ｜ 模型：`{engine_str}`")
    if df_backtest.empty or not backtest_ready:
        st.warning("請在左側上傳已完賽的結果 CSV。需有『賽事編號』、『馬號』及『名次』。")
    else:
        st.markdown("---")
        if st.button("🪄 一鍵 AI 參數尋優 (自動找出獨贏最高 ROI 參數)", type="primary", use_container_width=True):
            with st.spinner("🤖 AI 正在背景進行網格搜索 (Grid Search)，運算上百種參數組合..."):
                best_roi = -999
                best_params = None
                
                for test_ev in np.arange(1.0, 4.1, 0.5):
                    for test_odds in np.arange(2.0, 10.1, 1.0):
                        temp_invested = 0
                        temp_return = 0
                        bets_count = 0
                        
                        for race_id, group in df_backtest.groupby('賽事編號'):
                            sg = filter_and_sort_group(group, strategy_mode, test_ev, test_odds, max_odds, bet_type="win")
                            if len(sg) > 0:
                                pick = sg.iloc[0]
                                stake = max(10, float(pick.get('suggested_stake_win', 10)) if pd.notna(pick.get('suggested_stake_win')) else 10)
                                temp_invested += stake
                                bets_count += 1
                                
                                if pick['numeric_rank'] == 1:
                                    settlement_win_odds = pick.get('回測獨贏賠率', np.nan)
                                    if pd.isna(settlement_win_odds) or settlement_win_odds <= 1:
                                        settlement_win_odds = pick['獨贏賠率']
                                    temp_return += stake * float(settlement_win_odds)
                                    
                        if temp_invested > 0 and bets_count >= 2:
                            roi = ((temp_return - temp_invested) / temp_invested) * 100
                            if roi > best_roi:
                                best_roi = roi
                                best_params = {
                                    'ev': test_ev, 'odds': test_odds, 
                                    'bets': bets_count, 'roi': roi, 
                                    'profit': temp_return - temp_invested
                                }
                                
                if best_params and best_params['roi'] > 0:
                    st.success(f"🎉 **尋優完成！今日最佳破局密碼：**\n\n建議將左側滑桿設定為 ➡️ **EV 門檻 `{best_params['ev']:.1f}`** + **最低賠率 `{best_params['odds']:.1f}` 倍**。\n在此參數下，獨贏策略可出手 {best_params['bets']} 場，預期真實 **ROI 高達 {best_params['roi']:.2f}%** (淨利 ${best_params['profit']:.0f})！")
                else:
                    st.warning("⚠️ 在目前的策略與賽果下，找不到能產生穩定正利潤的參數組合。建議切換左側的「推薦排序核心邏輯」再跑一次尋優。")
        st.markdown("---")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "🥇 獨贏 (Win Engine)", "🥈 單匹位置 (Place Engine)", "🔗 位置Q (Place Engine)", "🥉 3匹位置包抄 (Place Engine)", "🛒 模擬結算 (Paper Trade)"
        ])

        with sub_tab1:
            st.markdown("#### 🥇 獨贏 (Win) 策略回測 — 由 `my_hkjc_model.pkl` 驅動")
            win_invested, win_return, win_bets, win_hits = 0, 0, 0, 0
            win_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_win_ev, min_odds, max_odds, bet_type="win")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    raw_stake = pick.get('suggested_stake_win', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    win_bets += 1
                    win_invested += stake
                    is_hit = (pick['numeric_rank'] == 1)
                    settlement_win_odds = pick.get('回測獨贏賠率', np.nan)
                    if pd.isna(settlement_win_odds) or settlement_win_odds <= 1:
                        settlement_win_odds = pick['獨贏賠率']
                        
                    payout = stake * float(settlement_win_odds) if is_hit else 0
                    if is_hit:
                        win_hits += 1
                        win_return += payout
                        result_str = "✅ 贏"
                    else:
                        result_str = "❌ 輸"
                        
                    win_records.append({
                        '賽事編號': race_id,
                        '賽道': pick.get('賽道', 'A'),
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        '買時賠率 (選馬用)': round(pick['獨贏賠率'], 1),
                        '最終賠率 (派彩用)': round(float(settlement_win_odds), 1),
                        'AI勝率': f"{pick['pred_win_prob']*100:.1f}%",
                        '買時 獨贏EV': round(pick['win_ev'], 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
            if win_bets > 0:
                roi = ((win_return - win_invested) / win_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{win_bets} 場")
                col2.metric("命中場數", f"{win_hits} 場", f"勝率: {win_hits/win_bets*100:.1f}%")
                col3.metric("總投入成本", f"${win_invested:,.0f}")
                col4.metric("總回收派彩", f"${win_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 獨贏明細 (對照買時賠率與最終派彩)")
                st.dataframe(pd.DataFrame(win_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合獨贏出手的場次。")

        with sub_tab2:
            st.markdown("#### 🥈 位置 (Place) 策略回測 — 由 `my_hkjc_place_model.pkl` 驅動")
            place_invested, place_return, place_bets, place_hits = 0, 0, 0, 0
            place_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    raw_stake = pick.get('suggested_stake_place', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    place_bets += 1
                    place_invested += stake
                    is_hit = (pick['numeric_rank'] <= 3)
                    settlement_place_odds = pick.get('回測位置賠率', np.nan)
                    if pd.isna(settlement_place_odds) or settlement_place_odds <= 1:
                        settlement_place_odds = pick['位置賠率']
                    p_odds = float(settlement_place_odds) if pd.notna(settlement_place_odds) and float(settlement_place_odds) > 1.0 else 1.5
                    
                    payout = stake * p_odds if is_hit else 0
                    if is_hit:
                        place_hits += 1
                        place_return += payout
                        result_str = "✅ 命中位置"
                    else:
                        result_str = "❌ 未入前三"
                        
                    place_records.append({
                        '賽事編號': race_id,
                        '賽道': pick.get('賽道', 'A'),
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        'AI入位率': f"{pick['pred_place_prob']*100:.1f}%",
                        '買時 位置EV': round(pick['place_ev'], 2),
                        '派彩賠率': round(p_odds, 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
            if place_bets > 0:
                roi = ((place_return - place_invested) / place_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{place_bets} 場")
                col2.metric("命中位置場數", f"{place_hits} 場", f"位置勝率: {place_hits/place_bets*100:.1f}%")
                col3.metric("總投入成本", f"${place_invested:,.0f}")
                col4.metric("總回收派彩", f"${place_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 位置明細")
                st.dataframe(pd.DataFrame(place_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合位置出手的場次。")

        with sub_tab3:
            st.markdown("#### 🔗 位置Q (QP) 策略回測 — 由 `my_hkjc_place_model.pkl` 驅動")
            qp_invested, qp_return, qp_bets, qp_hits = 0, 0, 0, 0
            qp_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")
                if len(sorted_group) >= 2:
                    top1 = sorted_group.iloc[0]
                    top2 = sorted_group.iloc[1]
                    stake = max(10, min(float(top1.get('suggested_stake_place',10) if pd.notna(top1.get('suggested_stake_place')) else 10), float(top2.get('suggested_stake_place',10) if pd.notna(top2.get('suggested_stake_place')) else 10)))
                    
                    qp_bets += 1
                    qp_invested += stake
                    is_hit = (top1['numeric_rank'] <= 3) and (top2['numeric_rank'] <= 3)
                    
                    p1_settlement = top1.get('回測位置賠率', np.nan)
                    p2_settlement = top2.get('回測位置賠率', np.nan)
                    p1_odds = float(p1_settlement if pd.notna(p1_settlement) and p1_settlement > 1.0 else top1['位置賠率'])
                    p2_odds = float(p2_settlement if pd.notna(p2_settlement) and p2_settlement > 1.0 else top2['位置賠率'])
                    estimated_qp_odds = round(p1_odds * p2_odds * 1.8, 1) 
                    
                    payout = stake * estimated_qp_odds if is_hit else 0
                    if is_hit:
                        qp_hits += 1
                        qp_return += payout
                        result_str = "✅ 命中位置Q"
                    else:
                        result_str = "❌ 落空"
                        
                    qp_records.append({
                        '賽事編號': race_id,
                        'QP 組合': f"{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})",
                        'AI入位率': f"{top1['pred_place_prob']*100:.0f}% / {top2['pred_place_prob']*100:.0f}%",
                        '實際名次': f"首選:第{str(top1['名次']).replace('.0','')}名 | 次選:第{str(top2['名次']).replace('.0','')}名",
                        '估算QP賠率': estimated_qp_odds,
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
            if qp_bets > 0:
                roi = ((qp_return - qp_invested) / qp_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{qp_bets} 場")
                col2.metric("命中 QP 場數", f"{qp_hits} 場", f"QP 命中率: {qp_hits/qp_bets*100:.1f}%")
                col3.metric("總投入成本", f"${qp_invested:,.0f}")
                col4.metric("總回收派彩", f"${qp_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 位置Q (QP) 明細")
                st.dataframe(pd.DataFrame(qp_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合位置 Q 出手的場次。")

        with sub_tab4:
            st.markdown("#### 🥉 3匹位置包抄 (3x Place) 策略回測 — 由 `my_hkjc_place_model.pkl` 驅動")
            st.caption("策略邏輯：每場買入位置模型推薦最高分的 3 匹馬位置。")
            p3_invested, p3_return, p3_bets, p3_hits = 0, 0, 0, 0
            p3_records = []
            
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")
                picks = sorted_group.head(3)
                
                for _, pick in picks.iterrows():
                    raw_stake = pick.get('suggested_stake_place', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    p3_bets += 1
                    p3_invested += stake
                    is_hit = (pick['numeric_rank'] <= 3)
                    
                    settlement_place_odds = pick.get('回測位置賠率', np.nan)
                    if pd.isna(settlement_place_odds) or settlement_place_odds <= 1:
                        settlement_place_odds = pick['位置賠率']
                    p_odds = float(settlement_place_odds) if pd.notna(settlement_place_odds) and float(settlement_place_odds) > 1.0 else 1.5
                    
                    payout = stake * p_odds if is_hit else 0
                    if is_hit:
                        p3_hits += 1
                        p3_return += payout
                        result_str = "✅ 命中位置"
                    else:
                        result_str = "❌ 未入前三"
                        
                    p3_records.append({
                        '賽事編號': race_id,
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        'AI入位率': f"{pick['pred_place_prob']*100:.1f}%",
                        '派彩賠率': round(p_odds, 2),
                        '買時 位置EV': round(pick['place_ev'], 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
                    
            if p3_bets > 0:
                roi = ((p3_return - p3_invested) / p3_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("總投注注數", f"{p3_bets} 注")
                col2.metric("命中注數", f"{p3_hits} 注", f"命中率: {p3_hits/p3_bets*100:.1f}%")
                col3.metric("總投入成本", f"${p3_invested:,.0f}")
                col4.metric("總回收派彩", f"${p3_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 3匹位置包抄明細")
                st.dataframe(pd.DataFrame(p3_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合出手的場次。")

        with sub_tab5:
            st.markdown("#### 🛒 模擬投注真實結算 (Paper Trading PnL)")
            st.caption("系統會將你的虛擬注單與官方賽果比對，並優先使用官方最終派彩賠率（若無則用買入賠率）計算真實盈虧。")
            
            if 'simulated_bets' not in st.session_state or st.session_state['simulated_bets'].empty:
                st.info("💡 目前沒有任何模擬投注紀錄。請先在「賽前預測」分頁進行模擬下注。")
            else:
                sim_df = st.session_state['simulated_bets'].copy()
                sim_results = []
                sim_invested, sim_return, sim_hits = 0, 0, 0
                sim_bets = len(sim_df)

                for idx, bet in sim_df.iterrows():
                    race_id = bet['賽事編號']
                    horse_no = str(bet['馬號'])
                    bet_type = bet['玩法']
                    locked_odds = float(bet['買入賠率'])
                    stake = float(bet['注碼'])

                    sim_invested += stake

                    match = df_backtest[(df_backtest['賽事編號'] == race_id) & (df_backtest['馬號'].astype(str) == horse_no)]

                    if match.empty or pd.isna(match['numeric_rank'].values[0]) or match['numeric_rank'].values[0] == 99:
                        sim_results.append({
                            '下注時間': bet['下注時間'],
                            '場次': bet['場次'],
                            '馬匹': f"{horse_no} ({bet['馬名']})",
                            '玩法': bet_type,
                            '注碼': f"${stake:.0f}",
                            '買入時賠率': locked_odds,
                            '實際名次': '-',
                            '結果': '⏳ 待開彩',
                            '派彩': '$0.0',
                            '淨盈虧': '$0.0'
                        })
                    else:
                        actual_rank = match['numeric_rank'].values[0]
                        rank_str = str(match['名次'].values[0]).replace('.0', '')

                        is_hit = False
                        if bet_type == "WIN" and actual_rank == 1:
                            is_hit = True
                            sp_odds = match.get('回測獨贏賠率', pd.Series([np.nan])).values[0]
                            final_odds = float(sp_odds) if pd.notna(sp_odds) and float(sp_odds) > 1.0 else locked_odds
                        elif bet_type == "PLA" and actual_rank <= 3:
                            is_hit = True
                            sp_odds = match.get('回測位置賠率', pd.Series([np.nan])).values[0]
                            final_odds = float(sp_odds) if pd.notna(sp_odds) and float(sp_odds) > 1.0 else locked_odds
                        else:
                            final_odds = locked_odds

                        if is_hit:
                            sim_hits += 1
                            payout = stake * final_odds
                            sim_return += payout
                            result_text = "✅ 贏"
                        else:
                            payout = 0
                            result_text = "❌ 輸"

                        sim_results.append({
                            '下注時間': bet['下注時間'],
                            '場次': bet['場次'],
                            '馬匹': f"{horse_no} ({bet['馬名']})",
                            '玩法': bet_type,
                            '注碼': f"${stake:.0f}",
                            '買入時賠率': locked_odds,
                            '最終派彩賠率': round(final_odds, 1),
                            '實際名次': f"第 {rank_str} 名",
                            '結果': result_text,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - stake:.1f}"
                        })

                if sim_invested > 0:
                    roi = ((sim_return - sim_invested) / sim_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("模擬總注數", f"{sim_bets} 注")
                    col2.metric("命中注數", f"{sim_hits} 注", f"命中率: {sim_hits/sim_bets*100:.1f}%")
                    col3.metric("總投入資金", f"${sim_invested:,.1f}")
                    col4.metric("總回收派彩", f"${sim_return:,.1f}", f"真實 ROI: {roi:.2f}%")

                    st.markdown("##### 🧾 模擬投注詳細結算單")
                    st.dataframe(pd.DataFrame(sim_results), use_container_width=True, hide_index=True)
