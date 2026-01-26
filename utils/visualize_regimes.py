"""
국면 분석 강화 시각화 (Plotly 인터랙티브)
국면별 특성, 전환 패턴, 성과 비교 등을 인터랙티브하게 시각화합니다.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict
import os


class RegimeVisualizer:
    """국면 시각화 클래스 (Plotly 기반)"""
    
    def __init__(self, results: Dict, regime_labels: np.ndarray, 
                 dates: pd.Series, returns: pd.DataFrame):
        """
        Args:
            results: 백테스트 결과
            regime_labels: 국면 레이블
            dates: 날짜
            returns: 수익률 DataFrame
        """
        self.results = results
        self.regime_labels = regime_labels
        self.dates = pd.to_datetime(dates)
        self.returns = returns
        self.weights = results['weights']
        
        # 색상 팔레트 (가독성 좋은 진한 색상)
        n_regimes = len(set(regime_labels))
        self.regime_colors = [
            '#1E88E5',  # 선명한 파랑
            '#E53935',  # 선명한 빨강
            '#43A047',  # 선명한 초록
            '#FB8C00',  # 선명한 오렌지
            '#8E24AA',  # 선명한 보라
        ][:n_regimes]
        
    def plot_regime_transitions(self, save_path: str = None):
        """국면 전환 매트릭스 (Plotly Heatmap)"""
        n_regimes = len(set(self.regime_labels))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(self.regime_labels) - 1):
            from_regime = self.regime_labels[i]
            to_regime = self.regime_labels[i + 1]
            transition_matrix[from_regime, to_regime] += 1
        
        # 정규화
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_prob = np.divide(transition_matrix, row_sums, 
                                   where=row_sums != 0, 
                                   out=np.zeros_like(transition_matrix))
        
        # Heatmap 생성
        labels = [f'Regime {i}' for i in range(n_regimes)]
        
        fig = go.Figure(data=go.Heatmap(
            z=transition_prob,
            x=labels,
            y=labels,
            colorscale='YlOrRd',
            text=np.round(transition_prob, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 14},
            colorbar=dict(title="Transition<br>Probability"),
            hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Regime Transition Probability Matrix',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title='To Regime',
            yaxis_title='From Regime',
            template='plotly_white',
            height=600,
            width=700
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Transition matrix saved: {save_path}")
        
        return fig
        
    def plot_regime_duration(self, save_path: str = None):
        """국면별 지속 기간 분포 (Plotly)"""
        durations = []
        current_regime = self.regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(self.regime_labels)):
            if self.regime_labels[i] == current_regime:
                current_duration += 1
            else:
                durations.append({
                    'regime': current_regime,
                    'duration': current_duration
                })
                current_regime = self.regime_labels[i]
                current_duration = 1
        
        durations.append({
            'regime': current_regime,
            'duration': current_duration
        })
        
        df_duration = pd.DataFrame(durations)
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Box Plot', 'Distribution'),
            horizontal_spacing=0.15
        )
        
        # 박스플롯
        for regime in sorted(df_duration['regime'].unique()):
            regime_data = df_duration[df_duration['regime'] == regime]['duration']
            fig.add_trace(
                go.Box(
                    y=regime_data,
                    name=f'Regime {regime}',
                    marker_color=self.regime_colors[regime],
                    boxmean='sd',
                    hovertemplate='Regime %{fullData.name}<br>Duration: %{y} weeks<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 히스토그램
        for regime in sorted(df_duration['regime'].unique()):
            regime_data = df_duration[df_duration['regime'] == regime]['duration']
            fig.add_trace(
                go.Histogram(
                    x=regime_data,
                    name=f'Regime {regime}',
                    marker_color=self.regime_colors[regime],
                    opacity=0.7,
                    nbinsx=20,
                    hovertemplate='Duration: %{x} weeks<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title={
                'text': 'Regime Duration Distribution',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            height=500,
            template='plotly_white',
            showlegend=True,
            barmode='overlay'
        )
        
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_yaxes(title_text="Duration (weeks)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (weeks)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Duration distribution saved: {save_path}")
        
        return fig
        
    def plot_regime_performance_comparison(self, save_path: str = None):
        """국면별 성과 비교 (Plotly)"""
        n_regimes = len(set(self.regime_labels))
        
        # 국면별 통계 계산
        regime_stats = []
        for regime in range(n_regimes):
            mask = self.regime_labels == regime
            regime_returns = (self.returns.iloc[mask].values * 
                            self.weights[mask]).sum(axis=1)
            
            regime_stats.append({
                'Regime': regime,
                'Mean Return (%)': regime_returns.mean() * 52 * 100,
                'Volatility (%)': regime_returns.std() * np.sqrt(52) * 100,
                'Sharpe Ratio': (regime_returns.mean() * 52) / 
                               (regime_returns.std() * np.sqrt(52)) 
                               if regime_returns.std() > 0 else 0,
                'Win Rate (%)': (regime_returns > 0).sum() / len(regime_returns) * 100,
                'Frequency (%)': mask.sum() / len(mask) * 100
            })
        
        df_stats = pd.DataFrame(regime_stats)
        
        # 2x2 서브플롯
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Annualized Return', 'Annualized Volatility', 
                          'Sharpe Ratio', 'Regime Frequency'),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 평균 수익률
        fig.add_trace(
            go.Bar(
                x=df_stats['Regime'],
                y=df_stats['Mean Return (%)'],
                marker_color=[self.regime_colors[i] for i in df_stats['Regime']],
                text=df_stats['Mean Return (%)'].round(2),
                textposition='outside',
                hovertemplate='Regime %{x}<br>Return: %{y:.2f}%<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 변동성
        fig.add_trace(
            go.Bar(
                x=df_stats['Regime'],
                y=df_stats['Volatility (%)'],
                marker_color=[self.regime_colors[i] for i in df_stats['Regime']],
                text=df_stats['Volatility (%)'].round(2),
                textposition='outside',
                hovertemplate='Regime %{x}<br>Volatility: %{y:.2f}%<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 샤프 비율
        fig.add_trace(
            go.Bar(
                x=df_stats['Regime'],
                y=df_stats['Sharpe Ratio'],
                marker_color=[self.regime_colors[i] for i in df_stats['Regime']],
                text=df_stats['Sharpe Ratio'].round(2),
                textposition='outside',
                hovertemplate='Regime %{x}<br>Sharpe: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 빈도
        fig.add_trace(
            go.Bar(
                x=df_stats['Regime'],
                y=df_stats['Frequency (%)'],
                marker_color=[self.regime_colors[i] for i in df_stats['Regime']],
                text=df_stats['Frequency (%)'].round(1),
                textposition='outside',
                hovertemplate='Regime %{x}<br>Frequency: %{y:.1f}%<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': 'Regime Performance Comparison',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            height=800,
            template='plotly_white'
        )
        
        # 축 업데이트
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_xaxes(title_text="Regime", row=1, col=2)
        fig.update_xaxes(title_text="Regime", row=2, col=1)
        fig.update_xaxes(title_text="Regime", row=2, col=2)
        
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Frequency (%)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Performance comparison saved: {save_path}")
        
        return df_stats, fig
    
    def plot_rolling_sharpe(self, window: int = 52, save_path: str = None):
        """롤링 샤프 비율 (국면별 색상, Plotly)"""
        portfolio_returns = (self.returns.values * self.weights).sum(axis=1)
        
        rolling_mean = pd.Series(portfolio_returns).rolling(window).mean() * 52
        rolling_std = pd.Series(portfolio_returns).rolling(window).std() * np.sqrt(52)
        rolling_sharpe = rolling_mean / rolling_std
        
        fig = go.Figure()
        
        # 국면별로 다른 색상으로 세그먼트 그리기
        for i in range(len(rolling_sharpe) - 1):
            if not np.isnan(rolling_sharpe.iloc[i]):
                regime = self.regime_labels[i]
                
                fig.add_trace(go.Scatter(
                    x=[self.dates.iloc[i], self.dates.iloc[i+1]],
                    y=[rolling_sharpe.iloc[i], rolling_sharpe.iloc[i+1]],
                    mode='lines',
                    line=dict(color=self.regime_colors[regime], width=2),
                    showlegend=False,
                    hovertemplate=f'Regime {regime}<br>Date: %{{x}}<br>Sharpe: %{{y:.2f}}<extra></extra>'
                ))
        
        # 범례용 더미 트레이스
        for regime in sorted(set(self.regime_labels)):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=self.regime_colors[regime], width=3),
                name=f'Regime {regime}'
            ))
        
        # 0선
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title={
                'text': f'{window}-Week Rolling Sharpe Ratio by Regime',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title='Date',
            yaxis_title='Rolling Sharpe Ratio',
            template='plotly_white',
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Rolling Sharpe saved: {save_path}")
        
        return fig
    
    def plot_3d_regime_space(self, save_path: str = None):
        """3D 국면 공간 시각화 (Plotly)"""
        # PCA 또는 t-SNE로 3D 축소
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(self.weights)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=[self.regime_colors[label] for label in self.regime_labels],
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=[f'Regime {label}<br>Date: {date.strftime("%Y-%m-%d")}' 
                  for label, date in zip(self.regime_labels, self.dates)],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': '3D Regime Space (PCA)',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
            ),
            template='plotly_white',
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ 3D regime space saved: {save_path}")
        
        return fig
    
    def generate_all_visualizations(self, save_dir: str):
        """모든 시각화 생성"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\nGenerating enhanced regime visualizations (Plotly)...")
        
        self.plot_regime_transitions(
            save_path=os.path.join(save_dir, 'regime_transitions.html')
        )
        
        self.plot_regime_duration(
            save_path=os.path.join(save_dir, 'regime_duration.html')
        )
        
        stats, _ = self.plot_regime_performance_comparison(
            save_path=os.path.join(save_dir, 'regime_performance.html')
        )
        
        self.plot_rolling_sharpe(
            save_path=os.path.join(save_dir, 'rolling_sharpe_by_regime.html')
        )
        
        self.plot_3d_regime_space(
            save_path=os.path.join(save_dir, 'regime_3d_space.html')
        )
        
        # 통계 저장
        stats.to_csv(os.path.join(save_dir, 'regime_statistics.csv'), index=False)
        print(f"✓ Regime statistics saved: {save_dir}/regime_statistics.csv")
        
        print(f"\n✅ All enhanced visualizations saved to {save_dir}")
        print(f"   All charts are interactive HTML files - open in browser!")
        
        return stats


if __name__ == "__main__":
    print("This script is meant to be imported and used with backtest results.")
    print("Example usage:")
    print("  from visualize_regimes import RegimeVisualizer")
    print("  visualizer = RegimeVisualizer(results, labels, dates, returns)")
    print("  visualizer.generate_all_visualizations('results/enhanced_plots/')")