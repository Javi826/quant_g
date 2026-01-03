    #ZX_BOT_operative.py
    """
    Módulo centralizado para cálculo de métricas de trading.
    Todas las métricas financieras del bot están aquí para evitar duplicación.
    """
    
    import pandas as pd
    import numpy as np
    from typing import Optional, Dict
    
    class BotState:
        def __init__(self):
            self.closed_total_profit = 0.0
    
    
    class MetricsCalculator:
        """
        Calculadora de métricas financieras para análisis de trading.
        Todas las funciones son estáticas y pueden usarse independientemente.
        """
        
        @staticmethod
        def profit_factor(df: pd.DataFrame) -> float:
            """
            Calcula el Profit Factor.
            
            Profit Factor = Total Wins / Total Losses
            
            Args:
                df: DataFrame con columna 'PROFIT'
            
            Returns:
                Profit Factor redondeado a 2 decimales
                Retorna 0 si no hay pérdidas
            """
            total_wins = df[df['PROFIT'] > 0]['PROFIT'].sum()
            total_losses = abs(df[df['PROFIT'] < 0]['PROFIT'].sum())
            
            if total_losses > 0:
                return round(total_wins / total_losses, 2)
            return 0.0
        
        @staticmethod
        def weekly_win_percentage(df: pd.DataFrame) -> float:
            """
            Calcula el porcentaje de semanas con profit positivo.
            
            Args:
                df: DataFrame con columnas 'PROFIT' y 'CLOSE_AT'
            
            Returns:
                Porcentaje de semanas positivas redondeado a 1 decimal
                Retorna 0 si no hay semanas
            """
            df = df.copy()
            df['CLOSE_DATE'] = pd.to_datetime(df['CLOSE_AT'])
            df['week'] = df['CLOSE_DATE'].dt.to_period('W')
            
            weekly_profit = df.groupby('week')['PROFIT'].sum()
            positive_weeks = len(weekly_profit[weekly_profit > 0])
            total_weeks = len(weekly_profit)
            
            if total_weeks > 0:
                return round((positive_weeks / total_weeks * 100), 1)
            return 0.0
        
        @staticmethod
        def max_drawdown_from_equity(equity_series: pd.Series) -> float:
            """
            Calcula el Max Drawdown en porcentaje desde una serie de equity.
            
            Args:
                equity_series: Serie de pandas con valores de equity
            
            Returns:
                Max Drawdown en % (valor negativo) redondeado a 2 decimales
                Retorna 0 si no hay drawdown
            """
            if len(equity_series) == 0:
                return 0.0
            
            peak = equity_series.cummax()
            drawdown = (equity_series - peak) / peak * 100
            
            return round(drawdown.min(), 2)
        
        @staticmethod
        def max_drawdown_from_trades(df: pd.DataFrame, capital_assigned: float) -> float:
            """
            Calcula el Max Drawdown directamente desde trades.
            
            Args:
                df: DataFrame con columnas 'PROFIT' y 'CLOSE_AT'
                capital_assigned: Capital asignado inicial
            
            Returns:
                Max Drawdown en % (valor negativo) redondeado a 2 decimales
            """
            if len(df) == 0 or capital_assigned == 0:
                return 0.0
            
            df = df.copy()
            df = df.sort_values('CLOSE_AT')
            
            cumulative_profit = df['PROFIT'].cumsum()
            equity = capital_assigned + cumulative_profit
            peak = equity.cummax()
            drawdown = (equity - peak) / peak * 100
            
            return round(drawdown.min(), 2)
        
        @staticmethod
        def sharpe_ratio(daily_returns: pd.Series) -> float:
            """
            Calcula el Sharpe Ratio anualizado.
            
            Sharpe = (Mean Return / Std Return) * sqrt(252)
            
            Args:
                daily_returns: Serie de retornos diarios (pct_change)
            
            Returns:
                Sharpe Ratio anualizado redondeado a 2 decimales
                Retorna 0 si no hay datos o std = 0
            """
            if len(daily_returns) == 0:
                return 0.0
            
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            
            if std_return > 0:
                return round((mean_return / std_return) * (252 ** 0.5), 2)
            return 0.0
        
        @staticmethod
        def ulcer_index(equity_series: pd.Series) -> float:
            """
            Calcula el Ulcer Index.
            
            Ulcer Index = sqrt(mean(drawdown_pct^2))
            
            Args:
                equity_series: Serie de pandas con valores de equity
            
            Returns:
                Ulcer Index redondeado a 2 decimales
                Retorna 0 si no hay datos
            """
            if len(equity_series) == 0:
                return 0.0
            
            peaks = equity_series.cummax()
            drawdown_pct = ((equity_series - peaks) / peaks) * 100
            ulcer = np.sqrt((drawdown_pct ** 2).mean())
            
            return round(ulcer, 2)
        
        @staticmethod
        def total_profit_usd(df: pd.DataFrame) -> float:
            """
            Calcula el profit total en USD.
            
            Args:
                df: DataFrame con columna 'PROFIT'
            
            Returns:
                Total profit redondeado a 2 decimales
            """
            return round(df['PROFIT'].sum(), 2)
        
        @staticmethod
        def total_profit_percentage(total_profit: float, capital_assigned: float) -> float:
            """
            Calcula el profit total en porcentaje.
            
            Args:
                total_profit: Profit total en USD
                capital_assigned: Capital asignado
            
            Returns:
                Profit % redondeado a 2 decimales
                Retorna 0 si capital_assigned es 0
            """
            if capital_assigned > 0:
                return round((total_profit / capital_assigned) * 100, 2)
            return 0.0
        
        @classmethod
        def calculate_all_metrics(cls, df: pd.DataFrame, capital_assigned: float, 
                                  include_profit_pct: bool = False) -> Dict[str, any]:
            """
            ✅ MÉTODO UNIFICADO - Calcula TODAS las métricas de forma consistente.
            Usado tanto por Curves como por Compose para garantizar métricas idénticas.
            
            Args:
                df: DataFrame con trades (columnas: PROFIT, CLOSE_AT)
                capital_assigned: Capital total asignado
                include_profit_pct: Si True, incluye total_profit_pct en el resultado (para Compose)
            
            Returns:
                Dict con todas las métricas + equity diaria (para gráficas en Curves)
            """
            if len(df) == 0:
                result = {
                    'num_trades': 0,
                    'total_profit_usd': 0.0,
                    'profit_factor': 0.0,
                    'weekly_win_pct': 0.0,
                    'win_rate': 0.0,
                    'max_dd': 0.0,
                    'ulcer_index': 0.0,
                    'sharpe_ratio': 0.0,
                    'daily_profit': pd.DataFrame()
                }
                if include_profit_pct:
                    result['total_profit_pct'] = 0.0
                return result
            
            # 1. Ordenar por fecha de cierre
            df_sorted = df.sort_values('CLOSE_AT').copy()
            
            # 2. Equity trade-by-trade (para Max DD preciso y Ulcer Index)
            cumulative_profit = df_sorted['PROFIT'].cumsum()
            equity_trades = capital_assigned + cumulative_profit
            
            # 3. Equity diaria (para Sharpe y gráficas)
            df_sorted['CLOSE_DATE'] = pd.to_datetime(df_sorted['CLOSE_AT'])
            df_sorted['date'] = df_sorted['CLOSE_DATE'].dt.date
            daily_profit = df_sorted.groupby('date')['PROFIT'].sum().reset_index()
            daily_profit['equity_usd'] = capital_assigned + daily_profit['PROFIT'].cumsum()
            
            # 4. Daily returns (usando pct_change de equity USD)
            daily_returns = daily_profit['equity_usd'].pct_change().dropna()
            
            # 5. Profit total
            total_profit = cls.total_profit_usd(df)
            
            # 6. Max DD (desde equity trade-by-trade para máxima precisión)
            max_dd = cls.max_drawdown_from_equity(equity_trades)
            
            # 7. Ulcer Index (desde equity trade-by-trade)
            ulcer = cls.ulcer_index(equity_trades)
            
            # 8. Número de trades
            num_trades = len(df)
            
            # 9. Win Rate (% de trades ganadores)
            win_rate = (len(df[df['PROFIT'] > 0]) / num_trades * 100) if num_trades > 0 else 0.0
            win_rate = round(win_rate, 1)
            
            # 10. Construir resultado
            result = {
                'num_trades': num_trades,
                'total_profit_usd': total_profit,
                'profit_factor': cls.profit_factor(df),
                'weekly_win_pct': cls.weekly_win_percentage(df),
                'win_rate': win_rate,
                'max_dd': max_dd,
                'ulcer_index': ulcer,
                'sharpe_ratio': cls.sharpe_ratio(daily_returns),
                'daily_profit': daily_profit  # ← Para Curves (gráficas)
            }
            
            if include_profit_pct:
                result['total_profit_pct'] = cls.total_profit_percentage(total_profit, capital_assigned)
            
            return result
    
    
    # ✅ Funciones de conveniencia para compatibilidad con código legacy
    def calculate_profit_factor(df: pd.DataFrame) -> float:
        """Wrapper para compatibilidad legacy"""
        return MetricsCalculator.profit_factor(df)
    
    
    def calculate_sharpe_ratio(daily_returns: pd.Series) -> float:
        """Wrapper para compatibilidad legacy"""
        return MetricsCalculator.sharpe_ratio(daily_returns)
    
    
    def calculate_max_drawdown(equity_series: pd.Series) -> float:
        """Wrapper para compatibilidad legacy"""
        return MetricsCalculator.max_drawdown_from_equity(equity_series)