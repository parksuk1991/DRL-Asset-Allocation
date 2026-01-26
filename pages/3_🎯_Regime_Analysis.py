"""
경로: pages/3_🎯_Regime_Analysis.py
국면 분석 페이지
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.clustering import RegimeDiscovery
from utils.visualize_regimes import RegimeVisualizer
from utils.session_state import init_session_state, get_temp_dir

# 페이지 설정
st.set_page_config(
    page_title="Regime Analysis - DRL Asset Allocation",
    page_icon="🎯",
    layout="wide"
)

# 세션 상태 초기화
init_session_state()

st.title("🎯 Regime Analysis")
st.markdown("---")

# 백테스트 결과 확인
if st.session_state.backtest_results is None:
    st.warning("⚠️ 먼저 Backtest 페이지에서 백테스트를 실행하세요.")
    st.stop()

st.success("✅ 백테스트 결과가 있습니다.")

# 국면 분석 설정
st.header("⚙️ 국면 분석 설정")

col1, col2 = st.columns(2)

with col1:
    st.subheader("클러스터링 설정")
    method = st.selectbox("클러스터링 방법", ["kmeans"], index=0)
    find_optimal = st.checkbox("최적 클러스터 수 자동 탐색", value=True)
    
    if find_optimal:
        min_clusters = st.slider("최소 클러스터 수", 2, 5, 3)
        max_clusters = st.slider("최대 클러스터 수", 3, 8, 5)
        n_clusters = None
    else:
        n_clusters = st.slider("클러스터 수", 2, 8, 3)
        min_clusters = n_clusters
        max_clusters = n_clusters

with col2:
    st.subheader("특징 선택")
    use_states = st.checkbox("상태 변수 포함", value=False, 
                            help="포트폴리오 비중뿐만 아니라 상태 변수도 클러스터링에 사용")
    
    st.info("""
    **국면 분석이란?**
    
    AI 에이전트의 포트폴리오 결정 패턴을 분석하여 
    시장 환경에 따른 투자 전략의 변화를 발견합니다.
    """)

# 국면 분석 실행
st.markdown("---")

if st.button("🚀 국면 분석 실행", type="primary", use_container_width=True):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 국면 분석 중...")
        progress_bar.progress(20)
        
        # 데이터 준비
        results = st.session_state.backtest_results
        data = st.session_state.processed_data
        states = st.session_state.states
        
        actions = results['weights']
        aligned_dates = results['dates'][:-1]
        
        if not isinstance(aligned_dates, pd.Series):
            aligned_dates = pd.Series(aligned_dates)
        
        # 인덱스 정렬
        aligned_indices = []
        for dt in aligned_dates:
            dt_ts = pd.Timestamp(dt)
            for i, d in enumerate(data['dates']):
                if pd.Timestamp(d) == dt_ts:
                    aligned_indices.append(i)
                    break
        
        regime_states = states[aligned_indices]
        aligned_returns = data['returns'].iloc[aligned_indices].reset_index(drop=True)
        
        progress_bar.progress(40)
        
        # 국면 발견기 생성
        if find_optimal:
            status_text.text("🔍 최적 클러스터 수 탐색 중...")
            temp_discoverer = RegimeDiscovery(
                method='kmeans',
                min_clusters=min_clusters,
                max_clusters=max_clusters,
            )
            optimal_k, scores = temp_discoverer.find_optimal_clusters(actions=actions)
            n_clusters = optimal_k
            st.info(f"✅ 최적 클러스터 수: {optimal_k}")
        
        progress_bar.progress(60)
        
        # 클러스터링 실행
        discoverer = RegimeDiscovery(
            method=method,
            n_clusters=n_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )
        
        if use_states:
            labels = discoverer.fit(actions=actions, states=regime_states)
        else:
            labels = discoverer.fit(actions=actions)
        
        progress_bar.progress(80)
        
        # 국면 분석
        status_text.text("📊 국면 통계 계산 중...")
        summary = discoverer.analyze_regimes(
            actions=actions,
            returns=aligned_returns,
            dates=aligned_dates,
        )
        
        progress_bar.progress(90)
        
        # 시각화
        status_text.text("🎨 시각화 생성 중...")
        visualizer = RegimeVisualizer(
            results={'weights': actions, 'portfolio_values': results['portfolio_values'][:-1]},
            regime_labels=labels,
            dates=aligned_dates,
            returns=aligned_returns
        )
        
        progress_bar.progress(100)
        status_text.text("✅ 국면 분석 완료!")
        
        # 세션에 저장
        st.session_state.regime_labels = labels
        st.session_state.regime_summary = summary
        st.session_state.regime_discoverer = discoverer
        st.session_state.regime_visualizer = visualizer
        
        st.balloons()
        st.success(f"✅ {n_clusters}개의 국면을 발견했습니다!")
        
    except Exception as e:
        st.error(f"❌ 국면 분석 중 오류 발생: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# 결과 표시
if st.session_state.regime_summary is not None:
    st.markdown("---")
    st.header("📊 국면 분석 결과")
    
    summary = st.session_state.regime_summary
    
    # 국면 요약 테이블
    st.subheader("📋 국면별 통계")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    
    # 국면별 주요 특징
    st.subheader("🔍 국면별 주요 특징")
    
    n_regimes = len(summary)
    cols = st.columns(min(n_regimes, 3))
    
    for idx, row in summary.iterrows():
        col_idx = idx % 3
        with cols[col_idx]:
            st.markdown(f"""
            **Regime {row['Regime_ID']}** ({row['Percentage']:.1f}% of time)
            
            - **Portfolio**
              - Return: {row['Annualized_Return(%)']:.2f}%
              - Vol: {row['Annualized_Volatility(%)']:.2f}%
            - **Weights**
              - S&P500: {row['S&P500_Mean']:.1%}
              - KOSPI: {row['KOSPI_Mean']:.1%}
              - Nikkei: {row['Nikkei_Mean']:.1%}
              - Euro: {row['Eurostoxx_Mean']:.1%}
            """)

# 시각화
st.markdown("---")
st.header("📈 시각화")

visualizer = st.session_state.regime_visualizer
discoverer = st.session_state.regime_discoverer

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "t-SNE", "국면 타임라인", "비중 분포", "전환 매트릭스", "지속 기간"
])

with tab1:
    st.subheader("t-SNE Visualization")
    fig = discoverer.plot_tsne()
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Regime Timeline")
    results = st.session_state.backtest_results
    fig = discoverer.plot_regimes(
        dates=pd.Series(results['dates'][:-1]),
        portfolio_values=results['portfolio_values'][:-1]
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Weight Distribution by Regime")
    fig = discoverer.plot_weight_distribution(results['weights'])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Regime Transition Matrix")
    fig = visualizer.plot_regime_transitions()
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Regime Duration Distribution")
    fig = visualizer.plot_regime_duration()
    st.plotly_chart(fig, use_container_width=True)

# 추가 분석
st.markdown("---")
st.header("📊 추가 분석")

tab1, tab2, tab3 = st.tabs([
    "성과 비교", "롤링 샤프", "3D 공간"
])

with tab1:
    st.subheader("Regime Performance Comparison")
    stats, fig = visualizer.plot_regime_performance_comparison()
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Rolling Sharpe by Regime")
    fig = visualizer.plot_rolling_sharpe(window=52)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("3D Regime Space (PCA)")
    fig = visualizer.plot_3d_regime_space()
    st.plotly_chart(fig, use_container_width=True)

# 엑셀 다운로드
st.markdown("---")
st.header("💾 결과 다운로드")

if st.button("📥 국면 분석 Excel 다운로드", use_container_width=True):
    try:
        temp_dir = get_temp_dir()
        excel_path = temp_dir / "regime_analysis.xlsx"
        
        discoverer.save_regime_summary_to_excel(summary, str(excel_path))
        
        with open(excel_path, 'rb') as f:
            st.download_button(
                label="⬇️ Excel 파일 다운로드",
                data=f,
                file_name="regime_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.success("✅ Excel 파일이 준비되었습니다!")
        
    except Exception as e:
        st.error(f"❌ Excel 생성 중 오류: {str(e)}")

       
