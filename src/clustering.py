"""
경로: src/clustering.py
사후 국면 발견 (Post-hoc Regime Discovery)
t-SNE + K-Means 기반 국면 발견 + Plotly 시각화 + 엑셀 저장
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Tuple
import os


class RegimeDiscovery:
    """t-SNE + K-Means 기반 국면 발견 (Plotly)"""

    def __init__(self, method: str = "kmeans", n_clusters: int = 3,
                 eps: float = 0.8, min_samples: int = 15,
                 min_clusters: int = 3, max_clusters: int = 5):
        self.method = method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.scaler = StandardScaler()
        self.labels = None
        self.features_for_tsne = None
        
        # Plotly 색상 (가독성 좋은 진한 색상)
        self.regime_colors = [
            '#1E88E5',  # 선명한 파랑
            '#E53935',  # 선명한 빨강
            '#43A047',  # 선명한 초록
            '#FB8C00',  # 선명한 오렌지
            '#8E24AA',  # 선명한 보라
        ]

    def fit(self, actions: np.ndarray, states: np.ndarray = None) -> np.ndarray:
        features = actions if states is None else np.concatenate([actions, states], axis=1)
        self.features_for_tsne = features.copy()

        print(f"\n{'='*60}")
        print(f"Performing {self.method.upper()} clustering...")
        print(f"Feature shape: {features.shape}")
        print(f"{'='*60}")

        features_scaled = self.scaler.fit_transform(features)

        clusterer = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=50,
            max_iter=500,
            random_state=42,
        )
        self.labels = clusterer.fit_predict(features_scaled)

        unique_labels = sorted(set(self.labels))
        if len(unique_labels) > 1:
            sil = silhouette_score(features_scaled, self.labels)
            cal = calinski_harabasz_score(features_scaled, self.labels)
            print(f"\n✓ Clustering completed!")
            print(f"  Clusters found: {len(unique_labels)}")
            print(f"  Silhouette Score: {sil:.4f}")
            print(f"  Calinski-Harabasz Score: {cal:.2f}")

        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nCluster Distribution:")
        for c, cnt in zip(unique, counts):
            pct = cnt / len(self.labels) * 100
            print(f"  Regime {c}: {cnt} samples ({pct:.1f}%)")
        return self.labels

    def find_optimal_clusters(self, actions: np.ndarray) -> Tuple[int, Dict]:
        """3~5개 범위에서 최적 k 선택"""
        features_scaled = self.scaler.fit_transform(actions)

        inertias, silhouettes, calinskis = [], [], []
        k_min = self.min_clusters
        k_max = min(self.max_clusters, len(actions) - 1)
        k_range = range(k_min, max(k_min + 1, k_max + 1))

        print("\nSearching for optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(features_scaled, labels))
            calinskis.append(calinski_harabasz_score(features_scaled, labels))
            print(f"  k={k}: Silhouette={silhouettes[-1]:.4f}, Calinski={calinskis[-1]:.2f}")

        best_idx = int(np.argmax(silhouettes))
        optimal_k = list(k_range)[best_idx]
        print(f"\n✓ Optimal number of clusters: {optimal_k}")

        return optimal_k, {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouettes": silhouettes,
            "calinskis": calinskis,
        }

    def analyze_regimes(self, actions: np.ndarray, returns: pd.DataFrame,
                        dates: pd.Series) -> pd.DataFrame:
        regime_stats = []
        unique_labels = sorted(set(self.labels))

        print("\n" + "=" * 80)
        print("REGIME ANALYSIS")
        print("=" * 80)

        for cluster in unique_labels:
            mask = self.labels == cluster
            if mask.sum() == 0:
                continue

            cluster_actions = actions[mask]
            cluster_returns = returns.iloc[mask]
            cluster_dates = pd.to_datetime(dates[mask])

            mean_w = cluster_actions.mean(axis=0)
            std_w = cluster_actions.std(axis=0)

            asset_ret_mean = cluster_returns.mean() * 52 * 100
            asset_ret_vol = cluster_returns.std() * np.sqrt(52) * 100

            port_ret = (cluster_returns.values * cluster_actions).sum(axis=1)
            mean_ret = port_ret.mean() * 52 * 100
            vol = port_ret.std() * np.sqrt(52) * 100
            sharpe = mean_ret / vol if vol > 0 else 0.0

            regime_stats.append({
                "Regime_ID": int(cluster),
                "Count": int(mask.sum()),
                "Percentage": mask.sum() / len(mask) * 100,
                "S&P500_Mean": mean_w[0],
                "KOSPI_Mean": mean_w[1],
                "Nikkei_Mean": mean_w[2],
                "Eurostoxx_Mean": mean_w[3],
                "S&P500_AnnRet(%)": asset_ret_mean.iloc[0],
                "KOSPI_AnnRet(%)": asset_ret_mean.iloc[1],
                "Nikkei_AnnRet(%)": asset_ret_mean.iloc[2],
                "Euro_AnnRet(%)": asset_ret_mean.iloc[3],
                "S&P500_AnnVol(%)": asset_ret_vol.iloc[0],
                "KOSPI_AnnVol(%)": asset_ret_vol.iloc[1],
                "Nikkei_AnnVol(%)": asset_ret_vol.iloc[2],
                "Euro_AnnVol(%)": asset_ret_vol.iloc[3],
                "Annualized_Return(%)": mean_ret,
                "Annualized_Volatility(%)": vol,
                "Sharpe_Ratio": sharpe,
            })

            print(f"\n  Regime ID={cluster} ({mask.sum()/len(mask)*100:.1f}% of time):")
            print(f"    Weights Mean - S&P500: {mean_w[0]:.1%}, "
                  f"KOSPI: {mean_w[1]:.1%}, "
                  f"Nikkei: {mean_w[2]:.1%}, "
                  f"Euro: {mean_w[3]:.1%}")
            print(f"    Portfolio: Return {mean_ret:.2f}%, "
                  f"Vol {vol:.2f}%, Sharpe {sharpe:.2f}")

        summary_df = pd.DataFrame(regime_stats)
        print("\n" + "=" * 80)
        print("REGIME ANALYSIS SUMMARY TABLE")
        print("=" * 80)
        print(summary_df)
        print("=" * 80)
        
        return summary_df

    def plot_all_regime_charts(self, actions: np.ndarray, dates: pd.Series,
                               portfolio_values: np.ndarray, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        print("\nGenerating regime charts (Plotly)...")
        self.plot_tsne(save_path=os.path.join(save_dir, "tsne_visualization.html"))
        self.plot_regimes(dates, portfolio_values,
                          save_path=os.path.join(save_dir, "regimes_timeline.html"))
        self.plot_weight_distribution(actions,
                                      save_path=os.path.join(save_dir, "weight_distribution.html"))
        print(f"✓ All regime charts saved to {save_dir}")

    def save_regime_summary_to_excel(self, summary_df: pd.DataFrame, save_path: str):
        """국면 분석 요약을 엑셀로 저장"""
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                base, ext = os.path.splitext(save_path)
                save_path = f"{base}_new{ext}"
                print(f"⚠ Original file is open. Saving as: {save_path}")
        
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Regime Summary", index=False)
        
        print(f"✓ Regime summary saved to Excel: {save_path}")

    def plot_tsne(self, save_path: str = None):
        """t-SNE 시각화 (Plotly)"""
        print("  Generating t-SNE visualization...")
        if self.features_for_tsne is None:
            return

        features_scaled = self.scaler.transform(self.features_for_tsne)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(features_scaled)

        # DataFrame 생성
        df_tsne = pd.DataFrame({
            'TSNE1': embedded[:, 0],
            'TSNE2': embedded[:, 1],
            'Regime': [f'Regime {label}' for label in self.labels]
        })

        fig = px.scatter(
            df_tsne,
            x='TSNE1',
            y='TSNE2',
            color='Regime',
            color_discrete_sequence=self.regime_colors,
            title='t-SNE: AI Agent Action/State Patterns',
            opacity=0.7,
            hover_data={'TSNE1': ':.2f', 'TSNE2': ':.2f'}
        )

        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        
        fig.update_layout(
            title={
                'text': 't-SNE: AI Agent Action/State Patterns',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            template='plotly_white',
            height=700,
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2'
        )

        if save_path:
            fig.write_html(save_path)
            print("    ✓ t-SNE plot saved")
        
        return fig

    def plot_regimes(self, dates: pd.Series, portfolio_values: np.ndarray,
                     save_path: str = None):
        """국면 타임라인 (Plotly)"""
        print("  Generating regime timeline...")

        dates_ts = pd.to_datetime(dates)
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value with Regime Labels', 'Regime Timeline'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )

        # 포트폴리오 가치
        fig.add_trace(
            go.Scatter(
                x=dates_ts,
                y=portfolio_values,
                mode='lines',
                line=dict(color='black', width=2),
                name='Portfolio',
                hovertemplate='Date: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # 국면별 포인트
        sorted_labels = sorted(set(self.labels))
        for label, color in zip(sorted_labels, self.regime_colors[:len(sorted_labels)]):
            mask = self.labels == label
            if mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=dates_ts[mask],
                        y=portfolio_values[mask],
                        mode='markers',
                        marker=dict(color=color, size=6, opacity=0.7),
                        name=f'Regime {label}',
                        hovertemplate=f'Regime {label}<br>Date: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
                    ),
                    row=1, col=1
                )

        # 국면 타임라인 (바 차트)
        for i in range(len(self.labels) - 1):
            label = self.labels[i]
            color_idx = sorted_labels.index(label)
            
            fig.add_shape(
                type="rect",
                x0=dates_ts.iloc[i],
                x1=dates_ts.iloc[i + 1],
                y0=0,
                y1=1,
                fillcolor=self.regime_colors[color_idx],
                opacity=0.7,
                line_width=0,
                row=2, col=1
            )

        fig.update_layout(
            title={
                'text': 'Portfolio Performance and Regime Timeline',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            template='plotly_white',
            height=900,
            hovermode='x unified',
            showlegend=True
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(title_text="Regime", row=2, col=1, range=[0, 1])

        if save_path:
            fig.write_html(save_path)
            print("    ✓ Regime timeline saved")
        
        return fig

    def plot_weight_distribution(self, actions: np.ndarray, save_path: str = None):
        """비중 분포 (Plotly)"""
        print("  Generating weight distribution...")

        asset_names = ["S&P500", "KOSPI200", "Nikkei225", "EuroStoxx50"]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=asset_names,
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        for i, asset in enumerate(asset_names):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            for label in sorted(set(self.labels)):
                mask = self.labels == label
                if mask.sum() == 0:
                    continue
                
                w = actions[mask, i]
                
                fig.add_trace(
                    go.Histogram(
                        x=w,
                        name=f'Regime {label}',
                        marker_color=self.regime_colors[label],
                        opacity=0.7,
                        nbinsx=20,
                        hovertemplate=f'Regime {label}<br>Weight: %{{x:.2%}}<br>Count: %{{y}}<extra></extra>',
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title={
                'text': 'Portfolio Weight Distribution by Regime',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            template='plotly_white',
            height=800,
            barmode='overlay'
        )

        # X축 레이블
        for i in range(1, 5):
            row = ((i-1) // 2) + 1
            col = ((i-1) % 2) + 1
            fig.update_xaxes(title_text="Weight", row=row, col=col, tickformat=',.0%')
            fig.update_yaxes(title_text="Frequency", row=row, col=col)

        if save_path:
            fig.write_html(save_path)
            print("    ✓ Weight distribution saved")
        

        return fig
