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

# 讀取模型
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
    
    # 確保有位置賠率欄位 (若無則自動估算)
    if '位置賠率' not in df_raw.columns:
        df_raw['位置賠率'] = 1.0 + (pd.to_numeric(df_raw.get('獨贏賠率', 10.0), errors='coerce') - 1.0) / 3.2
    
    st.subheader("⚡ 快速輸入臨場賠率")
    st.info("👇 修改下方表格的『獨贏賠率』或『位置賠率』，AI 會自動重新計算推薦與回測！")
    
    edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率', '位置賠率']
    df_editable = df_raw[edit_columns].copy()
    
    edited_df = st.data_editor(
        df_editable,
        disabled=['賽事編號', '馬號', '馬名', '排位檔位'],
        use_container_width=True,
        hide_index=True
    )
    
    df = df_raw.copy()
    df['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
    df['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce').fillna(1.5)
    
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
    
    # 建立 4 個主要大分頁，手機上一目了然
    tab_pred, tab_w, tab_p, tab_qp = st.tabs([
        "🎯 預測推薦", 
        "🥇 獨贏(W) 回測", 
        "🥈 位置(P) 回測", 
        "🔗 位置Q(QP) 回測"
    ])

    # 共用的對帳單上色函數
    def color_result(val):
        if '✅' in str(val): return 'color: #00FF00; font-weight: bold;'
        elif '❌' in str(val): return 'color: #FF4B4B;'
        return ''
        
    def color_profit(val):
        if isinstance(val, (int, float)):
            if val > 0: return 'color: #00FF00; font-weight: bold;'
            elif val < 0: return 'color: #FF4B4B;'
        return ''

    # ==========================================
    # 分頁 1：賽前預測推薦
    # ==========================================
    with tab_pred:
        st.subheader("🎯 LambdaRank 排序推薦清單")
        recommendations = []
        for race_id, group in df.groupby('賽事編號'):
            filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
            sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)

            if len(sorted_group) >= 2:
                top1 = sorted_group.iloc[0]
                top2 = sorted_group.iloc[1]
                top3 = sorted_group.iloc[2] if len(sorted_group) >= 3 else top2

                win_pick = f"馬號 {top1['馬號']} ({top1['馬名']}) [勝率:{top1['pred_win_prob']*100:.1f}%, EV:{top1['ev']:.2f}]"
                q_pick = f"{top1['馬號']} + {top2['馬號']}"
                qp_pick = f"{top1['馬號']} + {top2['馬號']} / {top3['馬號']}"

                recommendations.append({
                    '賽事編號': race_id,
                    '🎯 獨贏推薦': win_pick,
                    '🔗 Q / QP 推薦': q_pick + " | " + qp_pick
                })

        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.warning("⚠️ 沒有符合當前 EV 或賠率門檻的馬匹。")
        else:
            st.dataframe(rec_df, use_container_width=True)

    # 如果沒有真實名次，後面的回測就不顯示
    has_results = not (df['numeric_rank'] == 99).all()

    # ==========================================
    # 分頁 2：獨贏 (Win) 回測
    # ==========================================
    with tab_w:
        if not has_results:
            st.info("💡 目前上傳的 CSV 沒有真實名次，無法進行回測。")
        else:
            w_invested, w_return, w_bets, w_hits, w_bankroll = 0, 0, 0, 0, 0
            w_records = []
            
            for race_id, group in df.groupby('賽事編號'):
                filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)
                
                if len(sorted_group) > 0:
                    top1 = sorted_group.iloc[0]
                    w_bets += 1
                    w_invested += 100  # 每注 100 元
                    
                    is_win = (top1['numeric_rank'] == 1)
                    if is_win:
                        w_hits += 1
                        payout = 100 * top1['獨贏賠率']
                        result_str = "✅ 命中"
                    else:
                        payout = 0
                        result_str = "❌ 落空"
                        
                    net_profit = payout - 100
                    w_bankroll += net_profit
                    w_return += payout
                    
                    w_records.append({
                        '賽事編號': race_id,
                        '投注馬號': f"{top1['馬號']} ({top1['馬名']})",
                        '實際名次': int(top1['numeric_rank']) if top1['numeric_rank'] != 99 else "未知",
                        '獨贏賠率': top1['獨贏賠率'],
                        '結果': result_str,
                        '淨盈虧': net_profit,
                        '累積盈虧': w_bankroll
                    })
            
            if w_bets > 0:
                roi = ((w_return - w_invested) / w_invested) * 100
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("投注場數", f"{w_bets} 場")
                c2.metric("命中", f"{w_hits} 場", f"勝率: {w_hits/w_bets*100:.1f}%")
                c3.metric("總成本", f"${w_invested}")
                c4.metric("總回收", f"${w_return:.1f}", f"ROI: {roi:.2f}%")
                
                w_df = pd.DataFrame(w_records)
                st.line_chart(w_df[['賽事編號', '累積盈虧']].set_index('賽事編號'))
                styled_w = w_df.style.map(color_result, subset=['結果']).map(color_profit, subset=['淨盈虧', '累積盈虧']).format({"獨贏賠率": "{:.1f}", "淨盈虧": "${:.1f}", "累積盈虧": "${:.1f}"})
                st.dataframe(styled_w, use_container_width=True)
            else:
                st.info("💡 條件太嚴格，沒有獨贏出手場次。")

    # ==========================================
    # 分頁 3：位置 (Place) 回測
    # ==========================================
    with tab_p:
        if not has_results:
            st.info("💡 目前上傳的 CSV 沒有真實名次，無法進行回測。")
        else:
            p_invested, p_return, p_bets, p_hits, p_bankroll = 0, 0, 0, 0, 0
            p_records = []
            
            for race_id, group in df.groupby('賽事編號'):
                filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)
                
                if len(sorted_group) > 0:
                    top1 = sorted_group.iloc[0]
                    p_bets += 1
                    p_invested += 100
                    
                    # 位置條件：跑進前三名
                    is_place = (top1['numeric_rank'] <= 3)
                    p_odds = float(top1['位置賠率']) if pd.notna(top1['位置賠率']) else 1.5
                    
                    if is_place:
                        p_hits += 1
                        payout = 100 * p_odds
                        result_str = "✅ 命中位置"
                    else:
                        payout = 0
                        result_str = "❌ 未入三甲"
                        
                    net_profit = payout - 100
                    p_bankroll += net_profit
                    p_return += payout
                    
                    p_records.append({
                        '賽事編號': race_id,
                        '投注馬號': f"{top1['馬號']} ({top1['馬名']})",
                        '實際名次': int(top1['numeric_rank']) if top1['numeric_rank'] != 99 else "未知",
                        '位置賠率': p_odds,
                        '結果': result_str,
                        '淨盈虧': net_profit,
                        '累積盈虧': p_bankroll
                    })
            
            if p_bets > 0:
                roi = ((p_return - p_invested) / p_invested) * 100
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("投注場數", f"{p_bets} 場")
                c2.metric("命中位置", f"{p_hits} 場", f"勝率: {p_hits/p_bets*100:.1f}%")
                c3.metric("總成本", f"${p_invested}")
                c4.metric("總回收", f"${p_return:.1f}", f"ROI: {roi:.2f}%")
                
                p_df = pd.DataFrame(p_records)
                st.line_chart(p_df[['賽事編號', '累積盈虧']].set_index('賽事編號'))
                styled_p = p_df.style.map(color_result, subset=['結果']).map(color_profit, subset=['淨盈虧', '累積盈虧']).format({"位置賠率": "{:.2f}", "淨盈虧": "${:.1f}", "累積盈虧": "${:.1f}"})
                st.dataframe(styled_p, use_container_width=True)
            else:
                st.info("💡 條件太嚴格，沒有位置出手場次。")

    # ==========================================
    # 分頁 4：位置Q (QP) 回測
    # ==========================================
    with tab_qp:
        if not has_results:
            st.info("💡 目前上傳的 CSV 沒有真實名次，無法進行回測。")
        else:
            qp_invested, qp_return, qp_bets, qp_hits, qp_bankroll = 0, 0, 0, 0, 0
            qp_records = []
            
            for race_id, group in df.groupby('賽事編號'):
                filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                sorted_group = filtered_group.sort_values(by='raw_score', ascending=False).reset_index(drop=True)
                
                # QP 需要同場至少挑出 2 匹符合條件的馬
                if len(sorted_group) >= 2:
                    top1 = sorted_group.iloc[0]
                    top2 = sorted_group.iloc[1]
                    
                    qp_bets += 1
                    qp_invested += 100
                    
                    # QP條件：首選與次選雙雙跑進前三名
                    is_qp_hit = (top1['numeric_rank'] <= 3) and (top2['numeric_rank'] <= 3)
                    
                    # 位置Q 賠率簡單估算法 (兩匹馬位置賠率相乘 x 1.8)
                    p1_odds = float(top1['位置賠率']) if pd.notna(top1['位置賠率']) else 1.5
                    p2_odds = float(top2['位置賠率']) if pd.notna(top2['位置賠率']) else 1.5
                    est_qp_odds = round(p1_odds * p2_odds * 1.8, 1) 
                    
                    if is_qp_hit:
                        qp_hits += 1
                        payout = 100 * est_qp_odds
                        result_str = "✅ 命中位置Q"
                    else:
                        payout = 0
                        result_str = "❌ 落空"
                        
                    net_profit = payout - 100
                    qp_bankroll += net_profit
                    qp_return += payout
                    
                    qp_records.append({
                        '賽事編號': race_id,
                        'QP 組合': f"{top1['馬號']} + {top2['馬號']}",
                        '實際名次': f"首選:{int(top1['numeric_rank'])} | 次選:{int(top2['numeric_rank'])}",
                        '估算賠率': est_qp_odds,
                        '結果': result_str,
                        '淨盈虧': net_profit,
                        '累積盈虧': qp_bankroll
                    })
            
            if qp_bets > 0:
                roi = ((qp_return - qp_invested) / qp_invested) * 100
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("投注場數", f"{qp_bets} 場")
                c2.metric("命中 QP", f"{qp_hits} 場", f"勝率: {qp_hits/qp_bets*100:.1f}%")
                c3.metric("總成本", f"${qp_invested}")
                c4.metric("總回收", f"${qp_return:.1f}", f"ROI: {roi:.2f}%")
                
                qp_df = pd.DataFrame(qp_records)
                st.line_chart(qp_df[['賽事編號', '累積盈虧']].set_index('賽事編號'))
                styled_qp = qp_df.style.map(color_result, subset=['結果']).map(color_profit, subset=['淨盈虧', '累積盈虧']).format({"估算賠率": "{:.1f}", "淨盈虧": "${:.1f}", "累積盈虧": "${:.1f}"})
                st.dataframe(styled_qp, use_container_width=True)
            else:
                st.info("💡 目前設定下沒有符合位置 Q 出手的場次 (需同場至少有 2 匹馬達標)。")

else:
    st.info("👈 請在左側上傳今日賽前排位表 CSV 以啟動預測！")
