"""
경로: pages/4_💼_Production.py
실무 배포 페이지
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트를 Python 경로에 추가
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.backtesting import TrustRegionRebalancer
from utils.session_state import init_session_state, get_temp_dir

# 페이지 설정
st.set_page_config(
    page_title="Production - DRL Asset Allocation",
    page_icon="💼",
    layout="wide"
)

# 세션 상태 초기화
init_session_state()

st.title("💼 Production Deployment")
st.markdown("---")

# 모델 확인
if st.session_state.trained_model is None:
    st.warning("⚠️ 먼저 Training 페이지에서 모델을 학습하세요.")
    st.stop()

st.success("✅ 학습된 모델이 있습니다.")

# 최신 포트폴리오 비중 산출
st.header("📊 최신 포트폴리오 비중")

st.info("""
**실무 배포 시나리오**

학습된 모델을 사용하여 최신 데이터 기반으로 포트폴리오 비중을 산출합니다.
Look-Ahead Bias가 완전히 제거되어 실제 투자에 사용 가능합니다.
""")

if st.button("🚀 최신 비중 산출", type="primary", use_container_width=True):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📁 데이터 로드 중...")
        progress_bar.progress(20)
        
        # 데이터 로드
        temp_dir = get_temp_dir()
        data_path = temp_dir / "uploaded_data.csv"
        
        loader = DataLoader(data_path=str(data_path))
        data = loader.get_aligned_data()
        
        progress_bar.progress(40)
        
        # 특징 생성
        status_text.text("🔧 특징 생성 중 (Look-Ahead Bias 방지)...")
        engineer = FeatureEngineer(rolling_window=52)
        states, valid_indices = engineer.create_state_features(
            data['returns'],
            data['macro'],
            macro_lag=1  # 1기 lag 적용
        )
        
        progress_bar.progress(60)
        
        # 정규화
        train_size = int(len(valid_indices) * 0.7)
        train_indices = valid_indices[:train_size]
        states, _ = engineer.normalize_features(states, train_indices)
        
        progress_bar.progress(70)
        
        # 최신 상태 선택
        latest_idx = valid_indices[-1]
        latest_state = states[latest_idx]
        latest_date = data['dates'].iloc[latest_idx]
        
        status_text.text("🤖 모델 예측 중...")
        progress_bar.progress(80)
        
        # 모델 예측
        rebalancer = TrustRegionRebalancer(
            min_weight=0.05,
            max_weight=0.35,
            trust_region=0.15,
            action_scaling=1.5
        )
        
        raw_action = st.session_state.trained_model.predict(latest_state, deterministic=True)
        target_weights = rebalancer.action_to_weights(raw_action)
        
        progress_bar.progress(90)
        
        # 리스크 체크
        status_text.text("🔍 리스크 체크 중...")
        
        alerts = []
        
        # 집중도 체크
        hhi = np.sum(target_weights ** 2)
        if hhi > 0.35:
            alerts.append(f"⚠️ HIGH CONCENTRATION: HHI = {hhi:.3f} (>0.35)")
        
        # 극단적 비중 체크
        max_weight = np.max(target_weights)
        min_weight = np.min(target_weights)
        if max_weight > 0.5:
            alerts.append(f"⚠️ EXTREME WEIGHT: Max weight = {max_weight:.1%} (>50%)")
        if min_weight < 0.03:
            alerts.append(f"⚠️ VERY LOW WEIGHT: Min weight = {min_weight:.1%} (<3%)")
        
        # 데이터 신선도
        days_old = (datetime.now() - pd.to_datetime(latest_date)).days
        if days_old > 14:
            alerts.append(f"⚠️ DATA FRESHNESS: Data is {days_old} days old (>14 days)")
        
        progress_bar.progress(100)
        status_text.text("✅ 비중 산출 완료!")
        
        # 결과 저장
        asset_names = ['S&P500', 'KOSPI200', 'Nikkei225', 'EuroStoxx50']
        weights_dict = {name: float(w) for name, w in zip(asset_names, target_weights)}
        
        # 신뢰도 계산 (간단한 버전)
        train_states = states[valid_indices[:train_size]]
        mean_train_state = np.mean(train_states, axis=0)
        distance = np.linalg.norm(latest_state - mean_train_state)
        confidence = np.exp(-distance / 10)
        confidence = np.clip(confidence, 0, 1)
        
        result = {
            'date': str(latest_date),
            'weights': weights_dict,
            'confidence': float(confidence),
            'alerts': alerts,
            'data_freshness_days': days_old,
            'hhi': float(hhi)
        }
        
        st.session_state.production_result = result
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ 비중 산출 중 오류 발생: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# 결과 표시
if 'production_result' in st.session_state and st.session_state.production_result is not None:
    st.markdown("---")
    st.header("📋 포트폴리오 권장 사항")
    
    result = st.session_state.production_result
    
    # 기본 정보
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("기준 날짜", result['date'])
    with col2:
        st.metric("모델 신뢰도", f"{result['confidence']:.1%}")
    with col3:
        st.metric("데이터 신선도", f"{result['data_freshness_days']}일 전")
    
    # 포트폴리오 비중
    st.subheader("💼 권장 포트폴리오 비중")
    
    weights_df = pd.DataFrame([
        {'자산': k, '비중': f"{v:.2%}"} 
        for k, v in result['weights'].items()
    ])
    weights_df.loc[len(weights_df)] = {'자산': 'TOTAL', '비중': f"{sum(result['weights'].values()):.2%}"}
    
    st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    # 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        # 파이 차트
        import plotly.express as px
        fig = px.pie(
            values=list(result['weights'].values()),
            names=list(result['weights'].keys()),
            title='Portfolio Allocation'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 바 차트
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(
                x=list(result['weights'].keys()),
                y=list(result['weights'].values()),
                text=[f"{v:.1%}" for v in result['weights'].values()],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title='Portfolio Weights',
            yaxis_title='Weight',
            yaxis_tickformat=',.0%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 경고 및 리스크
    st.subheader("🚨 리스크 체크")
    
    if result['alerts']:
        for alert in result['alerts']:
            st.warning(alert)
    else:
        st.success("✅ 모든 리스크 체크를 통과했습니다.")
    
    # 추가 지표
    st.subheader("📊 포트폴리오 지표")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("HHI (집중도)", f"{result['hhi']:.3f}")
        hhi_status = "✅ 분산됨" if result['hhi'] <= 0.35 else "⚠️ 집중됨"
        st.caption(hhi_status)
    
    with col2:
        max_w = max(result['weights'].values())
        min_w = min(result['weights'].values())
        st.metric("비중 범위", f"{min_w:.1%} ~ {max_w:.1%}")
    
    # 실무 적용 가이드
    st.markdown("---")
    st.header("📝 실무 적용 가이드")
    
    # 신뢰도 기반 추천
    if result['confidence'] >= 0.7 and result['data_freshness_days'] <= 7:
        st.success("""
        **🟢 GREEN: 안전하게 적용 가능**
        
        - 모델 신뢰도가 높고 데이터가 신선합니다.
        - 제시된 비중을 직접 사용할 수 있습니다.
        """)
    elif result['confidence'] >= 0.5 and result['data_freshness_days'] <= 14:
        st.warning("""
        **🟡 YELLOW: 주의해서 적용**
        
        - 추가 리스크 오버레이를 고려하세요.
        - 시장 상황을 함께 점검하세요.
        """)
    else:
        st.error("""
        **🔴 RED: 추가 검토 필요**
        
        - 최신 데이터로 모델을 재학습하는 것을 권장합니다.
        - 리스크 관리팀과 협의하세요.
        """)
    
    # JSON 다운로드
    st.markdown("---")
    st.header("💾 결과 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON 다운로드
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 JSON 다운로드",
            data=json_str,
            file_name=f"portfolio_weights_{result['date']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # CSV 다운로드
        csv_data = pd.DataFrame([
            {'Date': result['date'], **result['weights']}
        ])
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv_data.to_csv(index=False),
            file_name=f"portfolio_weights_{result['date']}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("'최신 비중 산출' 버튼을 클릭하여 포트폴리오 비중을 생성하세요.")

# 모델 재학습 섹션
st.markdown("---")
st.header("🔄 모델 재학습")

st.info("""
**모델 재학습이 필요한 경우**

- 마지막 학습 이후 3개월이 경과한 경우
- 데이터가 14일 이상 오래된 경우
- 시장 환경이 크게 변화한 경우
""")

if st.button("🔄 모델 재학습 실행", use_container_width=True):
    st.warning("⚠️ 모델 재학습은 Training 페이지에서 수행할 수 있습니다.")
    st.info("👉 왼쪽 사이드바에서 **Training** 페이지로 이동하세요.")
