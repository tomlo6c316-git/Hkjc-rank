import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# 頁面基本設定
st.set_page_config(
    page_title="HKJC LambdaRank 賽馬預測系統",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 HKJC LambdaRank 智能排序系統")
st.markdown("---")

# 讀取模型 (請確保模型已上傳至同一個 GitHub 資料夾)
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

# 側邊欄設定
st.sidebar.header("📂 資料載入")
uploaded_file = st.sidebar.file_uploader("請上傳賽事資料 (CSV)", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 投注策略參數")
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
    
    st.subheader("⚡ 快速輸入臨場賠率 (支援手機直接點擊修改)")
    st.info("👇 點擊下方表格的『獨贏賠率』欄位即可修改數字。修改後，AI 會自動重新排序！")
    
    edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率']
    df_editable = df_raw[edit_columns].copy()
    
    edited_df = st.data_editor(
        df_editable,
        disabled=['賽事編號', '馬號', '馬名', '排位檔位'],
        use_container_width=True,
        hide_index=True
    )
    
    df = df_raw.copy()
    df['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
    
    # --- 特徵工程 ---
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

    # 取得 LambdaRank 實力分數
    df['raw_score'] = model.predict(X_predict)

    # Softmax 轉換為相對勝率
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    df['pred_win_prob'] = df.groupby('賽事編號')['raw_score'].transform(softmax)
    df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

    st.markdown("---")
    
    # 建立雙分頁
    tab1, tab2 = st.tabs(["🎯 排序預測推薦", "📈 歷史回測 (專業對帳單)"])

    # ====== 分頁 1：預測推薦 ======
    with tab1:
        st.subheader("🎯 LambdaRank 排序推薦清單")
        recommendations = []
        for race_id, group in df.groupby('賽事編號'):
            filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
            sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)

            if len(sorted_group) >= 2:
                top1 = sorted_group.iloc[0]
                top2 = sorted_group.iloc[1]
                top3 = sorted_group.iloc[2] if len(sorted_group) >= 3 else top2

                win_pick = f"馬號 {top1['馬號']} ({top1['馬名']}) [相對勝率:{top1['pred_win_prob']*100:.1f}%, EV:{top1['ev']:.2f}]"
                q_pick = f"{top1['馬號']} + {top2['馬號']}"
                qp_pick = f"{top1['馬號']} + {top2['馬號']} / {top3['馬號']}"

                recommendations.append({
                    '賽事編號': race_id,
                    '🎯 首選獨贏': win_pick,
                    '🔗 Q / QP': q_pick + " | " + qp_pick
                })

        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.warning("⚠️ 沒有符合當前 EV 或賠率門檻的馬匹。")
        else:
            st.dataframe(rec_df, use_container_width=True)

    # ====== 分頁 2：專業回測 ======
    with tab2:
        st.subheader("📊 排序模型策略回測結果 (專業對帳單)")
        
        if (df['numeric_rank'] == 99).all():
            st.info("💡 目前上傳的資料沒有真實名次，無法計算 ROI。請上傳包含完賽名次的歷史資料檔來執行回測。")
        else:
            total_invested = 0
            total_return = 0
            bet_count = 0
            win_count = 0
            backtest_records = []
            
            BET_AMOUNT = 100 
            current_bankroll = 0 
            
            for race_id, group in df.groupby('賽事編號'):
                filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)
                
                if len(sorted_group) > 0:
                    top1 = sorted_group.iloc[0]
                    bet_count += 1
                    total_invested += BET_AMOUNT
                    
                    is_win = (top1['numeric_rank'] == 1)
                    if is_win:
                        win_count += 1
                        payout = BET_AMOUNT * top1['獨贏賠率']
                        result_str = "✅ 命中"
                    else:
                        payout = 0
                        result_str = "❌ 落空"
                        
                    net_profit = payout - BET_AMOUNT
                    current_bankroll += net_profit
                    total_return += payout
                    
                    top2_info = f"馬號 {sorted_group.iloc[1]['馬號']} (跑第 {int(sorted_group.iloc[1]['numeric_rank'])} 名)" if len(sorted_group) > 1 else "-"
                    top3_info = f"馬號 {sorted_group.iloc[2]['馬號']} (跑第 {int(sorted_group.iloc[2]['numeric_rank'])} 名)" if len(sorted_group) > 2 else "-"
                    
                    backtest_records.append({
                        '賽事編號': race_id,
                        '🎯 首選馬 (獨贏)': f"馬號 {top1['馬號']} ({top1['馬名']})",
                        '預測勝率': f"{top1['pred_win_prob']*100:.1f}%",
                        '期望值 (EV)': top1['ev'],
                        '賠率': top1['獨贏賠率'],
                        '實際名次': int(top1['numeric_rank']) if top1['numeric_rank'] != 99 else "未知",
                        '結果': result_str,
                        '淨盈虧': net_profit,
                        '累積盈虧': current_bankroll,
                        '🥈 次選馬表現': top2_info,
                        '🥉 三選馬表現': top3_info
                    })
            
            if bet_count > 0:
                roi = ((total_return - total_invested) / total_invested) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{bet_count} 場")
                col2.metric("實際命中", f"{win_count} 場", f"勝率: {win_count/bet_count*100:.1f}%")
                col3.metric("總成本", f"${total_invested}")
                col4.metric("總回收", f"${total_return:.1f}", f"ROI: {roi:.2f}%")
                
                st.markdown("---")
                res_df = pd.DataFrame(backtest_records)
                
                st.markdown("#### 📈 累積盈虧走勢圖 (Bankroll Equity Curve)")
                chart_data = res_df[['賽事編號', '累積盈虧']].set_index('賽事編號')
                st.line_chart(chart_data)
                
                st.markdown("#### 📝 賽事詳細對帳單")
                
                def color_result(val):
                    if val == '✅ 命中': return 'color: #00FF00; font-weight: bold;'
                    elif val == '❌ 落空': return 'color: #FF4B4B;'
                    return ''
                    
                def color_profit(val):
                    if isinstance(val, (int, float)):
                        if val > 0: return 'color: #00FF00; font-weight: bold;'
                        elif val < 0: return 'color: #FF4B4B;'
                    return ''

                try:
                    styled_df = res_df.style.map(color_result, subset=['結果']).map(color_profit, subset=['淨盈虧', '累積盈虧']).format({"期望值 (EV)": "{:.2f}", "賠率": "{:.1f}", "淨盈虧": "${:.1f}", "累積盈虧": "${:.1f}"})
                except AttributeError:
                    styled_df = res_df.style.applymap(color_result, subset=['結果']).applymap(color_profit, subset=['淨盈虧', '累積盈虧']).format({"期望值 (EV)": "{:.2f}", "賠率": "{:.1f}", "淨盈虧": "${:.1f}", "累積盈虧": "${:.1f}"})
                
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("💡 在目前的 EV 和賠率篩選條件下，沒有任何場次符合出手標準。")
else:
    st.info("👈 請在左側上傳今日賽前排位表 CSV 以啟動預測！")
