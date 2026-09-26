import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import re
import time

# 頁面基本設定
st.set_page_config(page_title="HKJC 賽馬智能實戰系統", page_icon="🏆", layout="wide")
st.title("🏆 HKJC 賽馬智能實戰系統 (平注實戰版 + 即時賠率)")
st.markdown("---")

MODEL_PATH = 'my_hkjc_ranker.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"⚠️ 找不到 AI 模型檔 `{MODEL_PATH}`！請確認是否已上傳到 GitHub。")
    st.stop()

st.sidebar.header("📂 資料載入")
uploaded_file = st.sidebar.file_uploader("請上傳賽事資料 (CSV)", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("🎯 投注玩法設定")
bet_type = st.sidebar.radio(
    "選擇回測與預測玩法", 
    ["獨贏 (Win)", "位置 (Place) - 估算", "連贏 (Q) - 估算", "位置Q (QP) - 估算"]
)

bet_amount = st.sidebar.number_input("每場固定下注金額 ($)", min_value=10, value=100, step=10)
starting_bankroll = st.sidebar.number_input("初始資金庫 ($)", min_value=1000, value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 篩選門檻")
min_ev = st.sidebar.slider("最小期望值 (EV)", 0.0, 1.5, 0.0, 0.05)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=30.0)

# ==========================================
# 🌟 新增功能 1：抓取馬會即時賠率 API
# ==========================================
def fetch_live_odds(date_str, venue, race_no):
    """攔截馬會即時賠率 JSON"""
    url = f"https://bet.hkjc.com/racing/getJSON.aspx?type=winplaodds&date={date_str}&venue={venue}&raceno={race_no}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        # 萃取 獨贏=位置 結構
        matches = re.findall(r'(\d+)=(\d+\.\d+|\d+)=(\d+\.\d+|\d+)', resp.text)
        if matches:
            odds_dict = {}
            for m in matches:
                horse = str(m[0])
                win = float(m[1])
                odds_dict[horse] = {'win': win} # 這個版本專注於獨贏賠率
            return odds_dict
    except:
        pass
    return None

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding='cp950')
        
    st.success("✅ 賽事資料載入成功！")
    
    # ==========================================
    # 🌟 新增功能 2：使用 Session State 記憶 DataFrame
    # ==========================================
    if 'df_data_flat' not in st.session_state or st.session_state.get('uploaded_filename_flat') != uploaded_file.name:
        st.session_state['df_data_flat'] = df_raw.copy()
        st.session_state['uploaded_filename_flat'] = uploaded_file.name

    st.markdown("---")
    st.subheader("⚡ 臨場賠率更新中心")
    
    col_v, col_d, col_r, col_b = st.columns([1.5, 2, 1.5, 3])
    venue_input = col_v.selectbox("賽事場地", ["HV (跑馬地)", "ST (沙田)"])
    venue_code = "HV" if "HV" in venue_input else "ST"
    
    # 自動偵測日期
    sample_id = str(df_raw['賽事編號'].iloc[0])
    auto_date = f"{sample_id[:4]}-{sample_id[4:6]}-{sample_id[6:8]}" if len(sample_id) >= 8 else "2026-09-23"
    api_date = col_d.text_input("API 查詢日期", value=auto_date)
    
    # 偵測場次
    races_available = sorted(list(set([int(str(x).split('-')[1]) for x in df_raw['賽事編號']])))
    target_race = col_r.selectbox("更新場次", races_available)

    if col_b.button("🔄 一鍵抓取最新賠率 (開跑前 1 分鐘使用)", use_container_width=True):
        with st.spinner(f"正在連線馬會抓取第 {target_race} 場即時賠率..."):
            live_odds = fetch_live_odds(api_date, venue_code, target_race)
            
            if live_odds:
                df_temp = st.session_state['df_data_flat']
                race_mask = df_temp['賽事編號'].str.endswith(f"-{target_race:02d}")
                
                for horse_no, odds in live_odds.items():
                    horse_mask = race_mask & (df_temp['馬號'] == str(horse_no))
                    df_temp.loc[horse_mask, '獨贏賠率'] = odds['win']
                
                st.session_state['df_data_flat'] = df_temp
                st.success(f"✅ 第 {target_race} 場獨贏賠率更新成功！")
                time.sleep(1)
                st.rerun() 
            else:
                st.error("⚠️ 抓取失敗。可能是日期/場地錯誤，或馬會尚未開盤。")

    # ==========================================
    
    edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率']
    df_editable = st.session_state['df_data_flat'][edit_columns].copy()
    
    st.info("👇 賠率已同步更新。你也可以點擊表格手動修改，AI 將即時重新計算排序與勝率！")
    edited_df = st.data_editor(
        df_editable, disabled=['賽事編號', '馬號', '馬名', '排位檔位'],
        use_container_width=True, hide_index=True
    )
    
    df = st.session_state['df_data_flat'].copy()
    df['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
    
    # 特徵工程
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

    df['numeric_rank'] = pd.to_numeric(df.get('名次', 99), errors='coerce').fillna(99)
    df['jockey_win_rate'] = df.get('jockey_win_rate', 0.12)
    df['trainer_win_rate'] = df.get('trainer_win_rate', 0.12)
    df['combo_win_rate'] = df.get('combo_win_rate', 0.10)
    df['horse_win_rate'] = df.get('horse_win_rate', 0.10)
    df['horse_last_rank'] = df.get('horse_last_rank', 6.0)

    if '距離' not in df.columns: df['距離'] = 1200
    if 'horse_surface_win_rate' not in df.columns: df['horse_surface_win_rate'] = 0.08
    if 'horse_dist_win_rate' not in df.columns: df['horse_dist_win_rate'] = 0.08

    feature_cols = [
        'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
        '排位檔位', 'weight_diff', 'weight_rank',
        'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
        'horse_win_rate', 'horse_last_rank',
        '距離', 'horse_surface_win_rate', 'horse_dist_win_rate'
    ]

    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0
    X_predict = df[feature_cols].fillna(0)

    # 模型預測
    df['raw_score'] = model.predict(X_predict)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    df['pred_win_prob'] = df.groupby('賽事編號')['raw_score'].transform(softmax)
    df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

    st.markdown("---")
    tab1, tab2 = st.tabs(["🎯 今日預測推薦", "📈 平注歷史回測"])

    with tab1:
        st.subheader("🎯 AI 排序與實戰推薦名單")
        recommendations = []
        for race_id, group in df.groupby('賽事編號'):
            filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
            sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)

            if len(sorted_group) >= 2:
                top1 = sorted_group.iloc[0]
                top2 = sorted_group.iloc[1]
                
                win_pick = f"馬號 {top1['馬號']} ({top1['馬名']})"
                q_qp_pick = f"{top1['馬號']} + {top2['馬號']}"

                recommendations.append({
                    '賽事編號': race_id,
                    '🎯 獨贏/位置首選': win_pick,
                    '🔗 連贏/位置Q (雙馬)': q_qp_pick,
                    '📊 首選勝率': f"{top1['pred_win_prob']*100:.1f}%",
                    '💰 建議平注': f"${bet_amount}"
                })

        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.warning("⚠️ 沒有符合當前 EV 或賠率門檻的馬匹。")
        else:
            st.dataframe(rec_df, use_container_width=True)

    with tab2:
        st.subheader(f"📊 {bet_type} 平注回測結果")
        if (df['numeric_rank'] == 99).all():
            st.info("💡 目前上傳的資料沒有真實名次，無法執行回測。")
        else:
            total_invested = 0
            total_return = 0
            bet_count = 0
            win_count = 0
            backtest_records = []
            
            for race_id, group in df.groupby('賽事編號'):
                filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)
                
                if bet_type == "獨贏 (Win)" and len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    is_win = (pick['numeric_rank'] == 1)
                    odds = pick['獨贏賠率']
                    pick_str = f"{pick['馬號']}({pick['馬名']})"
                    
                elif bet_type == "位置 (Place) - 估算" and len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    is_win = (pick['numeric_rank'] <= 3) 
                    odds = max(1.05, pick['獨贏賠率'] / 3.5 + 0.3)
                    pick_str = f"{pick['馬號']}({pick['馬名']})"
                    
                elif bet_type == "連贏 (Q) - 估算" and len(sorted_group) >= 2:
                    pick1 = sorted_group.iloc[0]
                    pick2 = sorted_group.iloc[1]
                    is_win = (pick1['numeric_rank'] <= 2) and (pick2['numeric_rank'] <= 2)
                    odds = max(3.0, (pick1['獨贏賠率'] * pick2['獨贏賠率']) * 0.4)
                    pick_str = f"{pick1['馬號']} + {pick2['馬號']}"
                    
                elif bet_type == "位置Q (QP) - 估算" and len(sorted_group) >= 2:
                    pick1 = sorted_group.iloc[0]
                    pick2 = sorted_group.iloc[1]
                    is_win = (pick1['numeric_rank'] <= 3) and (pick2['numeric_rank'] <= 3)
                    odds = max(2.0, (pick1['獨贏賠率'] + pick2['獨贏賠率']) / 2.5)
                    pick_str = f"{pick1['馬號']} + {pick2['馬號']}"
                else:
                    continue
                
                bet_count += 1
                total_invested += bet_amount
                
                if is_win:
                    win_count += 1
                    payout = bet_amount * odds
                    total_return += payout
                    result_str = "✅ 命中"
                else:
                    payout = 0
                    result_str = "❌ 未命中"
                    
                backtest_records.append({
                    '賽事編號': race_id,
                    'AI 選擇': pick_str,
                    '下注金額': f"${bet_amount}",
                    '賠率': round(odds, 2),
                    '賽果': result_str,
                    '派彩金額': f"${payout:.1f}"
                })
            
            if len(backtest_records) > 0:
                total_profit = total_return - total_invested
                roi = (total_profit / total_invested * 100) if total_invested > 0 else 0.0
                final_bankroll = starting_bankroll + total_profit
                
                col1, col2, col3, col4 = col5 = st.columns(4)
                col1.metric("實際出手場數", f"{bet_count} 場")
                col2.metric("命中場數", f"{win_count} 場", f"勝率: {win_count/bet_count*100:.1f}%")
                col3.metric("總盈虧", f"${total_profit:.1f}", f"ROI: {roi:.2f}%")
                col4.metric("結算資金庫", f"${int(final_bankroll)}")
                
                st.markdown("---")
                st.dataframe(pd.DataFrame(backtest_records), use_container_width=True)
            else:
                st.info("💡 沒有任何場次符合出手標準。")
