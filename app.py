import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# 頁面基本設定
st.set_page_config(page_title="HKJC LambdaRank 賽馬預測系統", page_icon="🏆", layout="wide")
st.title("🏆 HKJC LambdaRank 智能排序系統 (整合凱利公式)")
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
st.sidebar.header("⚙️ 投注策略參數")
min_ev = st.sidebar.slider("最小期望值 (EV)", 0.0, 1.5, 0.0, 0.05)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=30.0)

# ==========================================
# 💰 新增：資金控管 (Bankroll Management)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("💰 凱利公式資金控管")
bankroll = st.sidebar.number_input("目前總本金 (Bankroll)", min_value=1000, value=10000, step=1000)
kelly_multiplier = st.sidebar.selectbox(
    "凱利比例 (建議保守)", 
    options=[0.1, 0.25, 0.5, 1.0], 
    index=1, # 預設選 0.25 (1/4 凱利)
    format_func=lambda x: f"{x*100}% 凱利 (1/{int(1/x)} Kelly)" if x != 1.0 else "100% 全凱利 (極度激進)"
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding='cp950')
        
    st.success("✅ 賽事資料載入成功！")
    st.subheader("⚡ 快速輸入臨場賠率")
    st.info("👇 點擊修改下方表格的『獨贏賠率』，AI 將即時計算最新勝率與【凱利建議注碼】！")
    
    edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率']
    df_editable = df_raw[edit_columns].copy()
    
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

    # 1. 取得 LambdaRank 分數並轉換為勝率
    df['raw_score'] = model.predict(X_predict)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    df['pred_win_prob'] = df.groupby('賽事編號')['raw_score'].transform(softmax)
    df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

    # ==========================================
    # 🧠 凱利公式核心運算
    # ==========================================
    # 淨賠率 (Net Odds)
    df['net_odds'] = df['獨贏賠率'] - 1.0 
    # 凱利比例 (Kelly Fraction) = p - (1-p)/b
    df['kelly_f'] = df['pred_win_prob'] - ((1.0 - df['pred_win_prob']) / df['net_odds'])
    # 如果期望值為負，凱利公式會算出負數，這時強制設為 0 (不下注)
    df['kelly_f'] = df['kelly_f'].clip(lower=0)
    
    # 最終建議下注金額 = 總本金 * 凱利比例 * 激進度參數
    # 四捨五入到十位數 (符合香港馬會最少 10 蚊一注的習慣)
    df['suggested_bet'] = (bankroll * df['kelly_f'] * kelly_multiplier).round(-1)

    st.markdown("---")
    tab1, tab2 = st.tabs(["🎯 預測推薦 & 凱利注碼", "📈 凱利動態回測 (ROI)"])

    with tab1:
        st.subheader("🎯 AI 推薦與精算注碼")
        recommendations = []
        for race_id, group in df.groupby('賽事編號'):
            filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds) & (group['suggested_bet'] > 0)]
            sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)

            if len(sorted_group) > 0:
                top1 = sorted_group.iloc[0]
                win_pick = f"馬號 {top1['馬號']} ({top1['馬名']})"
                
                recommendations.append({
                    '賽事編號': race_id,
                    '🎯 首選獨贏': win_pick,
                    '📊 相對勝率': f"{top1['pred_win_prob']*100:.1f}%",
                    '💰 建議下注金額': f"${int(top1['suggested_bet'])}",
                    '⚖️ 佔總本金比例': f"{(top1['suggested_bet']/bankroll)*100:.1f}%"
                })

        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.warning("⚠️ 沒有符合當前 EV 或賠率門檻的馬匹，凱利公式建議本場【袖手旁觀】。")
        else:
            st.dataframe(rec_df, use_container_width=True)

    with tab2:
        st.subheader("📊 凱利公式策略回測結果 (資金動態變化)")
        if (df['numeric_rank'] == 99).all():
            st.info("💡 目前上傳的資料沒有真實名次，無法執行回測。")
        else:
            # 這裡回測改成動態本金計算
            current_bankroll = bankroll
            bet_count = 0
            win_count = 0
            backtest_records = []
            
            for race_id, group in df.groupby('賽事編號'):
                # 重新動態計算這場的注碼 (基於當下的本金)
                group = group.copy()
                group['dynamic_bet'] = (current_bankroll * group['kelly_f'] * kelly_multiplier).round(-1)
                
                filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds) & (group['dynamic_bet'] > 0)]
                sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)
                
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    actual_bet = int(pick['dynamic_bet'])
                    
                    if actual_bet > 0:
                        bet_count += 1
                        current_bankroll -= actual_bet  # 先扣除下注金
                        
                        is_win = (pick['numeric_rank'] == 1)
                        if is_win:
                            win_count += 1
                            payout = actual_bet * pick['獨贏賠率']
                            current_bankroll += payout  # 贏了就把派彩加回本金
                            result_str = "✅ 命中"
                        else:
                            payout = 0
                            result_str = "❌ 未命中"
                            
                        backtest_records.append({
                            '賽事編號': race_id,
                            'AI 推薦馬號': f"馬號 {pick['馬號']} ({pick['馬名']})",
                            '賽前本金': f"${int(current_bankroll + actual_bet - payout)}",
                            '下注金額': f"${actual_bet}",
                            '獨贏賠率': pick['獨贏賠率'],
                            '結果': result_str,
                            '賽後本金': f"${int(current_bankroll)}"
                        })
            
            if bet_count > 0:
                roi = ((current_bankroll - bankroll) / bankroll) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("初始本金", f"${bankroll}")
                col2.metric("命中率", f"{win_count}/{bet_count} ({win_count/bet_count*100:.1f}%)")
                col3.metric("最終本金", f"${int(current_bankroll)}")
                col4.metric("總盈虧 (ROI)", f"{current_bankroll - bankroll:.1f}", f"{roi:.2f}%")
                
                st.markdown("---")
                st.dataframe(pd.DataFrame(backtest_records), use_container_width=True)
            else:
                st.info("💡 在目前的 EV 和賠率篩選條件下，沒有任何場次符合出手標準。")
