"""
경로: src/performance.py
성과 분석 모듈 (Plotly 인터랙티브 시각화)

개선사항:
1. Matplotlib → Plotly 전환
2. 인터랙티브 차트 (줌, 팬, 호버 정보)
3. 반응형 레이아웃
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict
import warnings
import os

warnings.filterwarnings("ignore")


class PerformanceAnalyzer:
    """성과 분석기 (Plotly 기반)"""

    def __init__(self, results: Dict, benchmark_returns: pd.Series, 
                 asset_returns: pd.DataFrame, asset_names: list):
        self.results = results
        self.benchmark_returns = benchmark_returns
        self.asset_returns = asset_returns
        self.asset_names = asset_names
        
        # Plotly 테마 설정
        self.colors = {
            'portfolio': '#1E88E5',      # 선명한 파랑
            'benchmark': '#E53935',      # 선명한 빨강
            'equal_weight': '#43A047',   # 선명한 초록
            'positive': '#43A047',
            'negative': '#E53935'
        }
        
        print(f"\n[PerformanceAnalyzer Initialization]")
        print(f"  Portfolio values length: {len(results['portfolio_values'])}")
        print(f"  Dates length: {len(results['dates'])}")

    def calculate_metrics(self) -> Dict:
        """성과 지표 계산"""
        portfolio_values = np.asarray(self.results["portfolio_values"])
        dates = pd.to_datetime(self.results["dates"])

        port_ret = np.diff(portfolio_values) / portfolio_values[:-1]
        ret_dates = dates.iloc[1:].reset_index(drop=True)

        bench_ret = self._align_benchmark_returns(ret_dates)
        eq_ret = self._calculate_equal_weight_returns(ret_dates)

        n = min(len(port_ret), len(bench_ret), len(eq_ret))
        if n == 0:
            return self._empty_metrics()

        port_ret = port_ret[:n]
        bench_ret = bench_ret[:n]
        eq_ret = eq_ret[:n]
        ret_dates = ret_dates.iloc[:n]

        port_vals = np.concatenate([[1.0], np.cumprod(1 + port_ret)])
        bench_vals = np.concatenate([[1.0], np.cumprod(1 + bench_ret)])
        eq_vals = np.concatenate([[1.0], np.cumprod(1 + eq_ret)])

        n_years = len(port_ret) / 52 if len(port_ret) > 0 else 0.0001

        port_metrics = self._calc_single_metrics(port_vals, n_years)
        bench_metrics = self._calc_single_metrics(bench_vals, n_years)
        eq_metrics = self._calc_single_metrics(eq_vals, n_years)

        ir_sp500 = self._calc_information_ratio(port_ret, bench_ret)
        ir_eq = self._calc_information_ratio(port_ret, eq_ret)
        te_sp500 = np.std(port_ret - bench_ret) * np.sqrt(52) * 100
        te_eq = np.std(port_ret - eq_ret) * np.sqrt(52) * 100

        return {
            "Portfolio": port_metrics,
            "S&P 500 (Benchmark)": bench_metrics,
            "Equal Weight": eq_metrics,
            "Information Ratio (vs S&P500)": ir_sp500,
            "Information Ratio (vs EqualWeight)": ir_eq,
            "Tracking Error vs S&P500 (%)": te_sp500,
            "Tracking Error vs EqualWeight (%)": te_eq,
        }

    def _calculate_equal_weight_returns(self, ret_dates: pd.Series) -> np.ndarray:
        """Equal Weight 수익률 계산"""
        ret_dates = pd.to_datetime(ret_dates)
        
        asset_ret = self.asset_returns.copy()
        if not isinstance(asset_ret.index, pd.DatetimeIndex):
            if len(asset_ret) == len(self.benchmark_returns):
                asset_ret.index = self.benchmark_returns.index
            else:
                return np.zeros(len(ret_dates))
        
        eq_weights = np.ones(asset_ret.shape[1]) / asset_ret.shape[1]
        eq_returns_full = (asset_ret.values * eq_weights).sum(axis=1)
        eq_series = pd.Series(eq_returns_full, index=asset_ret.index)
        
        eq_aligned = eq_series.reindex(ret_dates, method='ffill').fillna(0)
        
        return eq_aligned.values

    def _align_benchmark_returns(self, ret_dates: pd.Series) -> np.ndarray:
        """S&P500 수익률 정렬"""
        ret_dates = pd.to_datetime(ret_dates)
        bench = self.benchmark_returns.copy()
        bench_aligned = bench.reindex(ret_dates, method='ffill').fillna(0)
        return bench_aligned.values

    def _calc_single_metrics(self, values: np.ndarray, n_years: float) -> Dict:
        """단일 자산 지표 계산"""
        values = np.asarray(values)
        ret = np.diff(values) / values[:-1]

        total_ret = (values[-1] / values[0] - 1) * 100
        ann_ret = ((values[-1] / values[0]) ** (1 / n_years) - 1) * 100
        ann_vol = np.std(ret) * np.sqrt(52) * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        cummax = np.maximum.accumulate(values)
        dd = (values - cummax) / cummax
        max_dd = np.min(dd) * 100

        win_rate = np.sum(ret > 0) / len(ret) * 100 if len(ret) > 0 else 0.0

        return {
            "Total Return (%)": total_ret,
            "Annualized Return (%)": ann_ret,
            "Annualized Volatility (%)": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd,
            "Win Rate (%)": win_rate,
        }

    def _calc_information_ratio(self, ret: np.ndarray, bench_ret: np.ndarray) -> float:
        """정보 비율 계산"""
        n = min(len(ret), len(bench_ret))
        if n == 0:
            return 0.0
        ret = ret[:n]
        bench_ret = bench_ret[:n]

        excess = ret - bench_ret
        te = np.std(excess) * np.sqrt(52)
        if te > 0:
            return (np.mean(excess) * 52) / te
        return 0.0

    def _empty_metrics(self) -> Dict:
        """빈 지표 반환"""
        empty = {
            "Total Return (%)": 0.0,
            "Annualized Return (%)": 0.0,
            "Annualized Volatility (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
            "Win Rate (%)": 0.0,
        }
        return {
            "Portfolio": empty,
            "S&P 500 (Benchmark)": empty,
            "Equal Weight": empty,
            "Information Ratio (vs S&P500)": 0.0,
            "Information Ratio (vs EqualWeight)": 0.0,
            "Tracking Error vs S&P500 (%)": 0.0,
            "Tracking Error vs EqualWeight (%)": 0.0,
        }

    def print_metrics(self, metrics: Dict):
        """지표 출력"""
        print("\n" + "=" * 100)
        print("PERFORMANCE METRICS")
        print("=" * 100)
        print(f"\n{'Metric':<30} {'Portfolio':>15} {'S&P500':>15} {'EqualWeight':>15} {'vs S&P500':>15} {'vs EqW':>15}")
        print("-" * 100)

        port = metrics["Portfolio"]
        sp500 = metrics["S&P 500 (Benchmark)"]
        eq = metrics["Equal Weight"]

        for key in port.keys():
            p = port[key]
            s = sp500[key]
            e = eq[key]
            d_sp = p - s
            d_eq = p - e
            
            if "Ratio" in key or "Rate" in key:
                print(f"{key:<30} {p:>14.2f} {s:>14.2f} {e:>14.2f} {d_sp:>+14.2f} {d_eq:>+14.2f}")
            else:
                print(f"{key:<30} {p:>13.2f}% {s:>13.2f}% {e:>13.2f}% {d_sp:>+13.2f}% {d_eq:>+13.2f}%")

        print("-" * 100)
        print(f"{'Information Ratio (vs S&P500)':<30} {metrics['Information Ratio (vs S&P500)']:>14.2f}")
        print(f"{'Information Ratio (vs EqW)':<30} {metrics['Information Ratio (vs EqualWeight)']:>14.2f}")
        print(f"{'Tracking Error vs S&P500 (%)':<30} {metrics['Tracking Error vs S&P500 (%)']:>13.2f}%")
        print(f"{'Tracking Error vs EqW (%)':<30} {metrics['Tracking Error vs EqualWeight (%)']:>13.2f}%")
        print("=" * 100)

    def plot_all_charts(self, save_dir: str):
        """모든 차트 생성 (Plotly)"""
        os.makedirs(save_dir, exist_ok=True)
        print("\nGenerating interactive Plotly charts...")

        self.plot_weight_area_chart(
            save_path=os.path.join(save_dir, "portfolio_weights_area.html")
        )
        self.plot_performance_comparison(
            save_path=os.path.join(save_dir, "performance_comparison.html")
        )
        self.plot_drawdown(
            save_path=os.path.join(save_dir, "drawdown.html")
        )
        self.plot_rolling_metrics(
            save_path=os.path.join(save_dir, "rolling_metrics.html")
        )

        print(f"✓ All interactive charts saved to {save_dir}")

    def plot_weight_area_chart(self, save_path: str = None):
        """포트폴리오 비중 영역 차트 (Plotly)"""
        weights = np.asarray(self.results["weights"])
        dates = pd.to_datetime(self.results["dates"])
        plot_dates = dates.iloc[:-1]
        n = min(len(plot_dates), len(weights))
        
        fig = go.Figure()
        
        # 🎨 가독성 좋은 색상 (진한 색상으로 변경)
        colors_area = [
            '#2E5090',  # 진한 파랑 (S&P500)
            '#D64545',  # 진한 빨강 (KOSPI200)
            '#2CA02C',  # 진한 초록 (Nikkei225)
            '#FF8C00'   # 진한 오렌지 (EuroStoxx50) - 기존 연한 노란색 대체
        ]
        
        # 🔧 핵심 수정: groupnorm 제거, 값을 백분율로 직접 변환
        for i, (name, color) in enumerate(zip(self.asset_names, colors_area)):
            fig.add_trace(go.Scatter(
                x=plot_dates[:n],
                y=weights[:n, i] * 100,  # 0~1 값을 0~100으로 변환
                name=name,
                mode='lines',
                stackgroup='one',  # 누적 영역
                # groupnorm='percent' 제거! (이게 문제였음)
                fillcolor=color,
                line=dict(width=1, color=color),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x|%Y-%m-%d}<br>' +
                             'Weight: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': 'Portfolio Weight Allocation Over Time',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title='Date',
            yaxis_title='Portfolio Weight',
            yaxis=dict(
                ticksuffix='%',
                range=[0, 100],  # 0~100% 범위
                dtick=20  # 20% 간격으로 눈금
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Weight area chart saved: {save_path}")
        
        return fig

    def plot_performance_comparison(self, save_path: str = None):
        """성과 비교 차트 (Plotly)"""
        portfolio_values = np.asarray(self.results["portfolio_values"])
        dates = pd.to_datetime(self.results["dates"])

        port_ret = np.diff(portfolio_values) / portfolio_values[:-1]
        ret_dates = dates.iloc[1:].reset_index(drop=True)
        
        bench_ret = self._align_benchmark_returns(ret_dates)
        eq_ret = self._calculate_equal_weight_returns(ret_dates)

        n = min(len(port_ret), len(bench_ret), len(eq_ret))
        if n == 0:
            return

        port_ret = port_ret[:n]
        bench_ret = bench_ret[:n]
        eq_ret = eq_ret[:n]
        plot_dates = ret_dates.iloc[:n]

        port_cumval = np.cumprod(1 + port_ret) * 100
        bench_cumval = np.cumprod(1 + bench_ret) * 100
        eq_cumval = np.cumprod(1 + eq_ret) * 100

        fig = go.Figure()
        
        # 포트폴리오
        fig.add_trace(go.Scatter(
            x=plot_dates,
            y=port_cumval,
            name='DRL AI Agent Portfolio',
            line=dict(color=self.colors['portfolio'], width=3),
            hovertemplate='<b>DRL Portfolio</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         'Return: %{customdata:.2f}%<br>' +
                         '<extra></extra>',
            customdata=(port_cumval - 100)
        ))
        
        # S&P500
        fig.add_trace(go.Scatter(
            x=plot_dates,
            y=bench_cumval,
            name='S&P 500 Index',
            line=dict(color=self.colors['benchmark'], width=2, dash='dash'),
            hovertemplate='<b>S&P 500</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         'Return: %{customdata:.2f}%<br>' +
                         '<extra></extra>',
            customdata=(bench_cumval - 100)
        ))
        
        # Equal Weight
        fig.add_trace(go.Scatter(
            x=plot_dates,
            y=eq_cumval,
            name='Equal Weight Portfolio',
            line=dict(color=self.colors['equal_weight'], width=2, dash='dot'),
            hovertemplate='<b>Equal Weight</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         'Return: %{customdata:.2f}%<br>' +
                         '<extra></extra>',
            customdata=(eq_cumval - 100)
        ))
        
        fig.update_layout(
            title={
                'text': 'Cumulative Performance Comparison (Base = 100)',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title='Date',
            yaxis_title='Cumulative Return (Base=100)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Performance comparison saved: {save_path}")
        
        return fig

    def plot_drawdown(self, save_path: str = None):
        """낙폭 차트 (Plotly)"""
        portfolio_values = np.asarray(self.results["portfolio_values"])
        dates = pd.to_datetime(self.results["dates"])

        n = min(len(portfolio_values), len(dates))
        portfolio_values = portfolio_values[:n]
        dates = dates[:n]

        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax * 100

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.3)',
            line=dict(color='darkred', width=2),
            name='Drawdown',
            hovertemplate='<b>Drawdown</b><br>' +
                         'Date: %{x}<br>' +
                         'Drawdown: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Portfolio Drawdown Analysis',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            yaxis=dict(
                tickformat=',.1f'
            ),
            xaxis=dict(
                rangeslider=dict(visible=False)
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Drawdown chart saved: {save_path}")
        
        return fig
    
    def plot_rolling_metrics(self, window: int = 52, save_path: str = None):
        """롤링 지표 차트 (Plotly)"""
        portfolio_values = np.asarray(self.results["portfolio_values"])
        dates = pd.to_datetime(self.results["dates"])
        
        port_ret = np.diff(portfolio_values) / portfolio_values[:-1]
        ret_dates = dates.iloc[1:].reset_index(drop=True)
        
        # 롤링 샤프 비율
        rolling_mean = pd.Series(port_ret).rolling(window).mean() * 52
        rolling_std = pd.Series(port_ret).rolling(window).std() * np.sqrt(52)
        rolling_sharpe = rolling_mean / rolling_std
        
        # 롤링 변동성
        rolling_vol = pd.Series(port_ret).rolling(window).std() * np.sqrt(52) * 100
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio (52-week)', 'Rolling Volatility (52-week)'),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )
        
        # 샤프 비율
        fig.add_trace(
            go.Scatter(
                x=ret_dates,
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 0선
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        
        # 변동성
        fig.add_trace(
            go.Scatter(
                x=ret_dates,
                y=rolling_vol,
                name='Rolling Volatility',
                line=dict(color='#ff7f0e', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.2)',
                hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title={
                'text': 'Rolling Performance Metrics',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            height=800,
            template='plotly_white',
            hovermode='x unified',
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Rolling metrics saved: {save_path}")
        
        return fig

    def save_to_excel(self, save_path: str):
        """엑셀 저장"""
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                base, ext = os.path.splitext(save_path)
                save_path = f"{base}_new{ext}"
                print(f"⚠ Original file is open. Saving as: {save_path}")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            # Weights
            weights = np.asarray(self.results["weights"])
            dates = pd.to_datetime(self.results["dates"])
            weight_dates = dates.iloc[:-1]
            
            n = min(len(weight_dates), len(weights))
            weights_df = pd.DataFrame(
                weights[:n], 
                columns=self.asset_names, 
                index=weight_dates[:n]
            )
            weights_df.to_excel(writer, sheet_name="Portfolio Weights")

            # Values and Returns
            portfolio_values = np.asarray(self.results["portfolio_values"])
            port_ret = np.diff(portfolio_values) / portfolio_values[:-1]
            ret_dates = dates.iloc[1:].reset_index(drop=True)
            
            bench_ret = self._align_benchmark_returns(ret_dates)
            eq_ret = self._calculate_equal_weight_returns(ret_dates)
            
            n = min(len(port_ret), len(bench_ret), len(eq_ret))
            
            port_cumval = np.cumprod(1 + port_ret[:n])
            bench_cumval = np.cumprod(1 + bench_ret[:n])
            eq_cumval = np.cumprod(1 + eq_ret[:n])

            values_df = pd.DataFrame({
                "Date": ret_dates.iloc[:n],
                "Portfolio Value": port_cumval,
                "S&P500 Value": bench_cumval,
                "EqualWeight Value": eq_cumval,
                "Portfolio (Base=100)": port_cumval * 100,
                "S&P500 (Base=100)": bench_cumval * 100,
                "EqualWeight (Base=100)": eq_cumval * 100,
            })
            values_df.to_excel(writer, sheet_name="Portfolio Values", index=False)

            # Metrics
            metrics = self.calculate_metrics()
            metrics_data = []
            for key in ["Portfolio", "S&P 500 (Benchmark)", "Equal Weight"]:
                for metric_name, metric_value in metrics[key].items():
                    metrics_data.append({
                        "Category": key,
                        "Metric": metric_name,
                        "Value": metric_value
                    })
            
            metrics_data.extend([
                {"Category": "Relative", "Metric": "Information Ratio (vs S&P500)", 
                 "Value": metrics["Information Ratio (vs S&P500)"]},
                {"Category": "Relative", "Metric": "Information Ratio (vs EqualWeight)", 
                 "Value": metrics["Information Ratio (vs EqualWeight)"]},
                {"Category": "Relative", "Metric": "Tracking Error vs S&P500 (%)", 
                 "Value": metrics["Tracking Error vs S&P500 (%)"]},
                {"Category": "Relative", "Metric": "Tracking Error vs EqualWeight (%)", 
                 "Value": metrics["Tracking Error vs EqualWeight (%)"]},
            ])
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name="Performance Metrics", index=False)


        print(f"✓ Excel saved: {save_path}")
