import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class VisualEngineAgent:
    """
    Interactive Plotly Chart Generation Agent.
    Isolates UI rendering layouts from structural plot visual styling definitions.
    """
    def __init__(self, data: pd.DataFrame):
        self.df = data

    def plot_district_distribution(self) -> go.Figure:
        """Plots the macro aggregate camera deployment layout by administrative zone."""
        summary = self.df.groupby('Район')['Количество видеокамер'].sum().reset_index()
        summary = summary.sort_values(by='Количество видеокамер', ascending=True)
        
        fig = px.bar(
            summary, x='Количество видеокамер', y='Район', orientation='h',
            title='<b>Total Camera Deployment Volume by District</b>',
            labels={'Количество видеокамер': 'Camera Count', 'Район': 'Administrative District'},
            color='Количество видеокамер', color_continuous_scale='YlOrRd'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    def plot_street_type_distribution(self) -> go.Figure:
        """Generates distribution ratio of average cameras across environment zones."""
        summary = self.df.groupby('Street_Type')['Количество видеокамер'].mean().reset_index()
        summary = summary.sort_values(by='Количество видеокамер', ascending=False)
        
        fig = px.pie(
            summary, values='Количество видеокамер', names='Street_Type',
            title='<b>Average Safety Infrastructure Concentration Density</b>',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Sunset
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
