import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# 頁面基本設定
st.set_page_config(page_title="HKJC LambdaRank 賽馬預測系統", page_icon="🏆", layout="wide")
st.title("🏆 HKJC LambdaRank 智能排序系統 (全玩法支援)")
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
st.sidebar.header("🎯 投注玩法與策略")
# 🌟 新增：加入「連贏 (Q)」選項
bet_type = st.sidebar.radio(
    "選擇回測玩法", 
    ["獨贏 (Win)", "位置 (Place) - 估算", "連贏 (Q) - 估算", "位置Q (QP) - 估算"]
)

# 提示使用者
if bet_type != "獨贏 (Win)":
    st.sidebar.info("💡 因目前 CSV 只有獨贏賠率，位置、連贏與 QP 將強制使用 $100 平注進行勝率回測，派彩金額為系統經驗估算值。")
    betting_mode = "平注模式 (每場固定 $100)"
else:
    betting_mode = st.sidebar.radio("注碼模式 (僅限獨贏)", ["平注模式 (每場固定 $100)", "凱利公式 (動態注碼)"])

bankroll = st.sidebar.number_input("目前總本金 (Bankroll)", min_value=1000, value=10000, step=1000)
kelly_multiplier = st.sidebar.selectbox("凱利比例 (建議 0.25)", [0.1, 0.25, 0.5, 1.0], index=1)

# 篩選條件
min_ev = st.sidebar.slider("最小期望值 (EV)", 0.0, 1.5, 0.0, 0.05)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=30.0)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding='cp950')
        
    st.success("✅ 賽事資料載入成功！")
    
    edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率']
    df_editable = df_raw[edit_columns].copy()
    
    st.info("👇 點擊修改下方表格的『獨贏賠率』，AI 將即時重新計算排序與勝率！")
    edited_df = st.data_editor(
        df_editable, disabled=['賽事編號', '馬號', '馬名', '排位檔位'],
        use_container_width=True, hide_index=True
    )
    
    df = df_raw.copy()
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

    df['net_odds'] = df['獨贏賠率'] - 1.0 
    df['kelly_f'] = df['pred_win_prob'] - ((1.0 - df['pred_win_prob']) / df['net_odds'])
    df['kelly_f'] = df['kelly_f'].clip(lower=0)

    st.markdown("---")
    tab1, tab2 = st.tabs(["🎯 預測推薦清單", "📈 歷史回測 (命中率與盈虧)"])

    with tab1:
        st.subheader("🎯 AI 排序與策略推薦")
        recommendations = []
        for race_id, group in df.groupby('賽事編號'):
            filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
            sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)

            if len(sorted_group) >= 2:
                top1 = sorted_group.iloc[0]
                top2 = sorted_group.iloc[1]
                
                win_pick = f"馬號 {top1['馬號']} ({top1['馬名']})"
                q_qp_pick = f"{top1['馬號']} + {top2['馬號']}"
                
                if bet_type == "獨贏 (Win)":
                    base_bet = (bankroll * top1['kelly_f'] * kelly_multiplier)
                    suggested_bet = f"${int(round(base_bet, -1))}" if base_bet > 0 else "$0 (建議觀望)"
                else:
                    suggested_bet = "平注 $100"

                recommendations.append({
                    '賽事編號': race_id,
                    '🎯 獨贏/位置首選': win_pick,
                    '🔗 連贏/位置Q (雙馬)': q_qp_pick,
                    '📊 首選勝率': f"{top1['pred_win_prob']*100:.1f}%",
                    '💰 建議注碼': suggested_bet
                })

        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.warning("⚠️ 沒有符合當前 EV 或賠率門檻的馬匹。")
        else:
            st.dataframe(rec_df, use_container_width=True)

    with tab2:
        st.subheader(f"📊 {bet_type} 回測結果")
        if (df['numeric_rank'] == 99).all():
            st.info("💡 目前上傳的資料沒有真實名次，無法執行回測。")
        else:
            current_bankroll = bankroll
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
                    
                    if betting_mode == "平注模式 (每場固定 $100)":
                        actual_bet = 100
                    else:
                        kelly_f = max(0, pick['pred_win_prob'] - ((1.0 - pick['pred_win_prob']) / pick['net_odds']))
                        actual_bet = int(round(current_bankroll * kelly_f * kelly_multiplier, -1))
                        
                    pick_str = f"{pick['馬號']}({pick['馬名']})"
                    
                elif bet_type == "位置 (Place) - 估算" and len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    is_win = (pick['numeric_rank'] <= 3) 
                    odds = max(1.05, pick['獨贏賠率'] / 3.5 + 0.3)
                    actual_bet = 100
                    pick_str = f"{pick['馬號']}({pick['馬名']})"
                    
                # 🌟 新增：連贏 (Q) 的回測邏輯
                elif bet_type == "連贏 (Q) - 估算" and len(sorted_group) >= 2:
                    pick1 = sorted_group.iloc[0]
                    pick2 = sorted_group.iloc[1]
                    # 連贏條件：兩匹馬都必須在第 1 名或第 2 名
                    is_win = (pick1['numeric_rank'] <= 2) and (pick2['numeric_rank'] <= 2)
                    # 粗略估算連贏賠率 (約為兩馬獨贏賠率相乘的 40%)
                    odds = max(3.0, (pick1['獨贏賠率'] * pick2['獨贏賠率']) * 0.4)
                    actual_bet = 100
                    pick_str = f"{pick1['馬號']} + {pick2['馬號']}"
                    
                elif bet_type == "位置Q (QP) - 估算" and len(sorted_group) >= 2:
                    pick1 = sorted_group.iloc[0]
                    pick2 = sorted_group.iloc[1]
                    is_win = (pick1['numeric_rank'] <= 3) and (pick2['numeric_rank'] <= 3)
                    odds = max(2.0, (pick1['獨贏賠率'] + pick2['獨贏賠率']) / 2.5)
                    actual_bet = 100
                    pick_str = f"{pick1['馬號']} + {pick2['馬號']}"
                else:
                    actual_bet = 0
                
                # 只有實際注碼 > 0 且有挑選出馬匹才算一次真實出手
                if actual_bet > 0:
                    bet_count += 1
                    total_invested += actual_bet
                    
                    if is_win:
                        win_count += 1
                        payout = actual_bet * odds
                        result_str = "✅ 命中"
                    else:
                        payout = 0
                        result_str = "❌ 未命中"
                        
                    current_bankroll = current_bankroll - actual_bet + payout
                    total_return += payout
                    
                    backtest_records.append({
                        '賽事編號': race_id,
                        'AI 選擇': pick_str,
                        '下注金額': f"${actual_bet}",
                        '賠率 (獨贏或估算)': round(odds, 2),
                        '賽果': result_str,
                        '派彩金額': f"${payout:.1f}"
                    })
            
            if len(backtest_records) > 0:
                total_profit = total_return - total_invested
                roi_turnover = (total_profit / total_invested * 100) if total_invested > 0 else 0.0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("實際下注場數", f"{bet_count} 場")
                col2.metric("命中場數", f"{win_count} 場", f"勝率: {win_count/bet_count*100:.1f}%")
                col3.metric("總投注本金", f"${total_invested}")
                col4.metric("淨盈虧", f"${total_profit:.1f}", f"ROI: {roi_turnover:.2f}%")
                
                st.markdown("---")
                st.dataframe(pd.DataFrame(backtest_records), use_container_width=True)
            else:
                st.info("💡 沒有任何場次符合出手標準。")
