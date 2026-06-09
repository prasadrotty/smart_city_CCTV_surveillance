import pandas as pd
from typing import Dict, Any

class AnalyticsAgent:
    """
    Deep Descriptive & Macro Statistical Agent.
    Handles business-logic layer aggregations and computational summaries.
    """
    def __init__(self, data: pd.DataFrame):
        self.df = data

    def get_macro_metrics(self) -> Dict[str, Any]:
        """Calculates global structural baseline variables across the dataset."""
        return {
            "total_addresses": len(self.df),
            "total_cameras": int(self.df['Количество видеокамер'].sum()),
            "avg_cameras": float(self.df['Количество видеокамер'].mean()),
            "max_cameras": int(self.df['Количество видеокамер'].max()),
            "total_districts": int(self.df['Район'].nunique())
        }

    def get_district_leaderboard(self) -> pd.DataFrame:
        """Generates sorted, standardized infrastructure breakdowns per municipality district."""
        leaderboard = self.df.groupby('Район').agg(
            total_cameras=('Количество видеокамер', 'sum'),
            installation_nodes=('Адрес', 'count'),
            density_per_node=('Количество видеокамер', 'mean')
        ).reset_index()
        
        return leaderboard.sort_values(by='total_cameras', ascending=False)
