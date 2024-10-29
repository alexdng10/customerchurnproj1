import plotly.graph_objects as go

def create_gauge_chart(churn_risk):
    # Set color scheme based on risk level with blue and pink tones
    gauge_color = "deepskyblue" if churn_risk < 0.3 else "mediumorchid" if churn_risk < 0.6 else "hotpink"

    # Construct gauge indicator with new color scheme
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=churn_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Customer Churn Risk", 'font': {'size': 22, 'color': 'deepskyblue'}},
            number={'font': {'size': 36, 'color': 'mediumorchid'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'lightpink'},
                'bar': {'color': gauge_color},
                'bgcolor': "rgba(30, 30, 60, 0.5)",
                'borderwidth': 1,
                'bordercolor': 'lightpink',
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 191, 255, 0.4)'},       # Blue for low risk
                    {'range': [30, 60], 'color': 'rgba(218, 112, 214, 0.4)'},   # Orchid for medium risk
                    {'range': [60, 100], 'color': 'rgba(255, 105, 180, 0.4)'}    # Pink for high risk
                ],
                'threshold': {
                    'line': {'color': "aqua", 'width': 3},
                    'thickness': 0.75,
                    'value': churn_risk * 100
                }
            }
        )
    )

    # Layout adjustments with blue and pink tones
    gauge_fig.update_layout(
        paper_bgcolor="rgba(10, 10, 30, 0.9)",
        plot_bgcolor="rgba(10, 10, 30, 0.9)",
        font={'color': 'lightpink'},
        width=420,
        height=320,
        margin=dict(l=15, r=15, t=40, b=20)
    )

    return gauge_fig

def create_model_probability_chart(model_probabilities):
    model_names = list(model_probabilities.keys())
    risk_scores = list(model_probabilities.values())

    # Horizontal bar chart for model risk scores with blue and pink bars
    risk_fig = go.Figure(data=[
        go.Bar(
            x=risk_scores, y=model_names, orientation='h',
            text=[f'{score:.1%}' for score in risk_scores],
            textposition='auto',
            marker=dict(color='rgba(255, 105, 180, 0.8)')  # Hot pink bars for models
        )
    ])

    # Adjust chart layout with new blue and pink theme
    risk_fig.update_layout(
        title='Model-Specific Churn Risk',
        yaxis_title='Models',
        xaxis_title='Churn Probability',
        xaxis=dict(tickformat='.1%', range=[0, 1], showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='rgba(10, 10, 30, 0.9)',   # Dark blue background
        paper_bgcolor='rgba(10, 10, 30, 0.9)',
        font={'color': 'deepskyblue'},
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return risk_fig
