"""
백테스트 페이지
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.backtesting import Backtester, TrustRegionRebalancer
from src.performance import PerformanceAnalyzer
from utils.session_state import init_session_state, get_temp_dir

# 페이지 설정
st.set_page_config(
    page_title="Backtest - DRL Asset Allocation",
    page_icon="📈",
    layout="wide"
)

# 세션 상태 초기화
init_session_state()

st.title("📈 Backtest Analysis")
st.markdown("---")

# 모델 확인
if st.session_state.trained_model is None:
    st.warning("⚠️ 먼저 Training 페이지에서 모델을 학습하세요.")
    st.stop()

st.success("✅ 학습된 모델이 있습니다.")

# 백테스트 설정
st.header("⚙️ 백테스트 설정")

col1, col2 = st.columns(2)

with col1:
    st.subheader("리밸런싱 제약")
    min_weight = st.slider("최소 비중", 0.0, 0.2, 0.05, 0.01)
    max_weight = st.slider("최대 비중", 0.2, 0.6, 0.35, 0.05)
    trust_region = st.slider("Trust Region", 0.05, 0.3, 0.15, 0.05)

with col2:
    st.subheader("거래 설정")
    transaction_cost = st.number_input("거래 비용", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
    rebalance_freq = st.selectbox("리밸런싱 주기 (주)", [1, 2, 4, 8, 12], index=2)

# 백테스트 실행
st.markdown("---")

if st.button("🚀 백테스트 실행", type="primary", use_container_width=True):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 백테스트 실행 중...")
        progress_bar.progress(20)
        
        # Rebalancer 생성
        rebalancer = TrustRegionRebalancer(
            min_weight=min_weight,
            max_weight=max_weight,
            trust_region=trust_region,
            action_scaling=1.5
        )
        
        progress_bar.progress(40)
        
        # Backtester 생성
        data = st.session_state.processed_data
        backtester = Backtester(
            returns=data['returns'],
            dates=data['dates'],
            rebalancer=rebalancer,
            transaction_cost=transaction_cost,
            rebalance_freq=rebalance_freq,
        )
        
        progress_bar.progress(60)
        
        # 백테스트 실행
        test_indices = st.session_state.test_indices
        results = backtester.run(
            agent=st.session_state.trained_model,
            states=st.session_state.states,
            start_idx=test_indices[0],
            end_idx=test_indices[-1] + 1,
        )
        
        progress_bar.progress(80)
        
        # 성과 분석
        analyzer = PerformanceAnalyzer(
            results=results,
            benchmark_returns=data['sp500_returns'],
            asset_returns=data['returns'],
            asset_names=['S&P500', 'KOSPI200', 'Nikkei225', 'EuroStoxx50'],
        )
        
        metrics = analyzer.calculate_metrics()
        
        progress_bar.progress(100)
        status_text.text("✅ 백테스트 완료!")
        
        # 세션에 저장
        st.session_state.backtest_results = results
        st.session_state.performance_metrics = metrics
        st.session_state.performance_analyzer = analyzer
        
        st.balloons()
        st.success("✅ 백테스트가 완료되었습니다!")
        
    except Exception as e:
        st.error(f"❌ 백테스트 중 오류 발생: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# 결과 표시
if st.session_state.backtest_results is not None:
    st.markdown("---")
    st.header("📊 성과 지표")
    
    metrics = st.session_state.performance_metrics
    
    # 주요 지표 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        port_return = metrics['Portfolio']['Annualized Return (%)']
        bench_return = metrics['S&P 500 (Benchmark)']['Annualized Return (%)']
        st.metric(
            "연간 수익률",
            f"{port_return:.2f}%",
            f"{port_return - bench_return:+.2f}% vs S&P500"
        )
    
    with col2:
        port_vol = metrics['Portfolio']['Annualized Volatility (%)']
        bench_vol = metrics['S&P 500 (Benchmark)']['Annualized Volatility (%)']
        st.metric(
            "연간 변동성",
            f"{port_vol:.2f}%",
            f"{port_vol - bench_vol:+.2f}% vs S&P500"
        )
    
    with col3:
        port_sharpe = metrics['Portfolio']['Sharpe Ratio']
        bench_sharpe = metrics['S&P 500 (Benchmark)']['Sharpe Ratio']
        st.metric(
            "샤프 비율",
            f"{port_sharpe:.2f}",
            f"{port_sharpe - bench_sharpe:+.2f} vs S&P500"
        )
    
    with col4:
        port_mdd = metrics['Portfolio']['Max Drawdown (%)']
        st.metric(
            "최대 낙폭",
            f"{port_mdd:.2f}%"
        )
    
    # 상세 지표 테이블
    st.subheader("📋 상세 성과 비교")
    
    comparison_data = []
    for metric_name in ['Total Return (%)', 'Annualized Return (%)', 
                       'Annualized Volatility (%)', 'Sharpe Ratio', 
                       'Max Drawdown (%)', 'Win Rate (%)']:
        port_val = metrics['Portfolio'][metric_name]
        sp500_val = metrics['S&P 500 (Benchmark)'][metric_name]
        eq_val = metrics['Equal Weight'][metric_name]
        
        comparison_data.append({
            '지표': metric_name,
            'Portfolio': f"{port_val:.2f}",
            'S&P500': f"{sp500_val:.2f}",
            'Equal Weight': f"{eq_val:.2f}",
            'vs S&P500': f"{port_val - sp500_val:+.2f}",
            'vs EqW': f"{port_val - eq_val:+.2f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # 추가 지표
    col1, col2 = st.columns(2)
    with col1:
        st.metric("정보 비율 (vs S&P500)", f"{metrics['Information Ratio (vs S&P500)']:.2f}")
    with col2:
        st.metric("추적 오차 (vs S&P500)", f"{metrics['Tracking Error vs S&P500 (%)']:.2f}%")
    
    # 차트 섹션
    st.markdown("---")
    st.header("📈 시각화")
    
    analyzer = st.session_state.performance_analyzer
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "포트폴리오 비중", "누적 성과", "낙폭 분석", "롤링 지표"
    ])
    
    with tab1:
        st.subheader("Portfolio Weight Allocation")
        fig = analyzer.plot_weight_area_chart()
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Cumulative Performance Comparison")
        fig = analyzer.plot_performance_comparison()
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Drawdown Analysis")
        fig = analyzer.plot_drawdown()
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Rolling Metrics")
        fig = analyzer.plot_rolling_metrics()
        st.plotly_chart(fig, use_container_width=True)
    
    # 엑셀 다운로드
    st.markdown("---")
    st.header("💾 결과 다운로드")
    
    if st.button("📥 Excel 다운로드", use_container_width=True):
        try:
            temp_dir = get_temp_dir()
            excel_path = temp_dir / "backtest_results.xlsx"
            
            analyzer.save_to_excel(str(excel_path))
            
            with open(excel_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Excel 파일 다운로드",
                    data=f,
                    file_name="portfolio_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            st.success("✅ Excel 파일이 준비되었습니다!")
            
        except Exception as e:
            st.error(f"❌ Excel 생성 중 오류: {str(e)}")

else:
    st.info("백테스트를 실행하면 결과가 여기에 표시됩니다.")
