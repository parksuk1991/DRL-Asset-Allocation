"""
경로: src/data_loader.py
데이터 로딩 및 전처리 모듈 (벤치마크 수정)
BBG_data.csv 파일을 읽고 기본적인 전처리를 수행합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class DataLoader:
    """Bloomberg data loader"""
    
    def __init__(self, data_path: str = "data/BBG_data.csv"):
        """
        Args:
            data_path: CSV file path
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.index_data = None
        self.macro_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Read CSV file and return as DataFrame"""
        print(f"Loading data from {self.data_path}...")
        
        # Read CSV file
        try:
            self.raw_data = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.raw_data = pd.read_csv(self.data_path, encoding='cp949')
        
        # Convert column A to datetime
        self.raw_data.iloc[:, 0] = pd.to_datetime(self.raw_data.iloc[:, 0])
        self.raw_data = self.raw_data.sort_values(by=self.raw_data.columns[0])
        self.raw_data = self.raw_data.reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.raw_data)} rows")
        print(f"  Date range: {self.raw_data.iloc[0, 0]} to {self.raw_data.iloc[-1, 0]}")
        
        return self.raw_data
    
    def split_data(self) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Split data into dates, indices, and macro variables
        
        Returns:
            dates: Column A (dates)
            indices: Columns B~E (S&P500, KOSPI, Nikkei, Eurostoxx)
            macros: From column F, 17 macro variables
        """
        if self.raw_data is None:
            self.load_data()
        
        # Dates (column A)
        dates = self.raw_data.iloc[:, 0]
        
        # Index prices (columns B~E: 4 indices)
        self.index_data = self.raw_data.iloc[:, 1:5].copy()
        self.index_data.columns = ['S&P500', 'KOSPI200', 'Nikkei225', 'EuroStoxx50']
        
        # Macro variables (from column F, 17 variables)
        self.macro_data = self.raw_data.iloc[:, 5:22].copy()
        
        print(f"✓ Data split completed")
        print(f"  Indices shape: {self.index_data.shape}")
        print(f"  Macro variables shape: {self.macro_data.shape}")
        
        return dates, self.index_data, self.macro_data
    
    def calculate_returns(self, window: int = 1) -> pd.DataFrame:
        """
        Calculate index returns
        
        Args:
            window: Return calculation window (1 = weekly return)
            
        Returns:
            Returns DataFrame
        """
        if self.index_data is None:
            self.split_data()
        
        # Calculate log returns
        returns = np.log(self.index_data / self.index_data.shift(window))
        
        # Remove first window rows (NaN)
        returns = returns.iloc[window:].reset_index(drop=True)
        
        print(f"✓ Returns calculated")
        print(f"  Returns shape: {returns.shape}")
        
        return returns
    
    def get_aligned_data(self) -> Dict[str, pd.DataFrame]:
        """
        Return complete data aligned after calculating returns
        
        Returns:
            dict: {
                'returns': returns DataFrame,
                'macro': macro DataFrame,
                'dates': dates Series,
                'sp500_returns': S&P500 returns Series (for benchmark)
            }
        """
        dates, index_data, _ = self.split_data()
        returns = self.calculate_returns()
        
        # Align returns and macro data (remove first row)
        aligned_macro = self.macro_data.iloc[1:].reset_index(drop=True)
        aligned_dates = dates.iloc[1:].reset_index(drop=True)
        
        # Remove NaN
        valid_idx = ~(returns.isna().any(axis=1) | aligned_macro.isna().any(axis=1))
        
        result = {
            'returns': returns[valid_idx].reset_index(drop=True),
            'macro': aligned_macro[valid_idx].reset_index(drop=True),
            'dates': aligned_dates[valid_idx].reset_index(drop=True),
            # S&P500 수익률을 날짜 인덱스로 저장 (핵심 수정)
            'sp500_returns': pd.Series(
                returns['S&P500'][valid_idx].values,
                index=pd.to_datetime(aligned_dates[valid_idx]),
                name='S&P500'
            )
        }
        
        print(f"\n✓ Aligned data prepared")
        print(f"  Total length: {len(result['returns'])}")
        print(f"  Date range: {result['dates'].iloc[0]} to {result['dates'].iloc[-1]}")
        print(f"  S&P500 returns range: {result['sp500_returns'].min():.4f} to {result['sp500_returns'].max():.4f}")
        print(f"  S&P500 returns mean: {result['sp500_returns'].mean():.6f}")
        
        return result


if __name__ == "__main__":
    # Test code
    loader = DataLoader()
    data = loader.get_aligned_data()
    
    print("\n=== Data Summary ===")
    print(f"Returns:\n{data['returns'].describe()}")
    print(f"\nMacro:\n{data['macro'].describe()}")

    print(f"\nS&P500 Returns:\n{data['sp500_returns'].describe()}")
