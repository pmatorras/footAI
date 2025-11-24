"""
Model visualization utilities for footAI (Plotly version).
Generates feature importance charts and confusion matrices from training results.
"""

import json
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from typing import Optional

def generate_model_visualizations(json_path, output_dir='figures/model_viz', top_n=15):
    """
    Generate both feature importance and confusion matrix plots.
    Shared function used by both train.py and plot.py.
    
    Parameters
    ----------
    json_path : str or Path
        Path to model results JSON
    output_dir : str or Path
        Directory to save visualizations
    top_n : int
        Number of top features to display
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    #try:
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nGenerating visualizations...")
    
    # Feature importance
    #plot_feature_importance(str(json_path), top_n=top_n, output_path=str(output_dir / f'{json_path.stem}_feature_importance.png'))
    
    # Confusion matrix
    plot_confusion_matrix(str(json_path),output_path=str(output_dir / f'{json_path.stem}_confusion_matrix.png'))
    
    print(f"Visualizations saved to {output_dir}/")
    return True
    '''
    except Exception as e:
        print(f"WARNING!  Could not generate visualizations: {e}")
        return False
    '''

def plot_feature_importance(json_path: str, top_n: int = 15, output_path: Optional[str] = None) -> go.Figure:
    """
    Plot feature importance from model training results using Plotly.
    
    Parameters
    ----------
    json_path : str
        Path to model results JSON file
    top_n : int, default=15
        Number of top features to display
    output_path : str, optional
        If provided, save figure to this path (HTML or PNG)
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract feature importance
    features = data['feature_importance'][:top_n]
    feature_names = [f['feature'] for f in features]
    importance_values = [f['importance'] * 100 for f in features]  # Convert to percentage
    
    # Create figure with color gradient (top 3 highlighted)
    colors = ['coral' if i < 3 else 'steelblue' for i in range(len(feature_names))]
    
    fig = go.Figure(go.Bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        marker=dict(color=colors, opacity=0.8),
        text=[f'{v:.2f}%' for v in importance_values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>'
    ))
    
    # Reverse y-axis so highest importance is at top
    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(
        title=dict(
            text=f'Top {top_n} Feature Importance - {_extract_model_name(json_path)}',
            font=dict(size=16, family='Arial', color='#333')
        ),
        xaxis_title='Importance (%)',
        yaxis_title='',
        template='plotly_white',
        height=max(400, top_n * 25),  # Scale height with number of features
        margin=dict(l=150, r=80, t=80, b=60),
        hovermode='y unified'
    )
    print("output path", output_path)
    if output_path:
        if output_path.endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_image(output_path, width=1000, height=max(400, top_n * 25))
        print(f"Saved feature importance plot to {output_path}")
    
    return fig


def plot_confusion_matrix(json_path: str, output_path: Optional[str] = None ) -> go.Figure:
    """
    Plot confusion matrix from model training results using Plotly.
    
    Parameters
    ----------
    json_path : str
        Path to model results JSON file
    output_path : str, optional
        If provided, save figure to this path (HTML or PNG)
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract confusion matrix
    cm = np.array(data['confusion_matrix']['matrix'])
    labels = data['confusion_matrix']['labels']
    
    # Calculate percentages for annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create hover text
    hover_text = []
    for i in range(len(labels)):
        hover_row = []
        for j in range(len(labels)):
            hover_row.append(
                f'True: {labels[i]}<br>Predicted: {labels[j]}<br>'
                f'Count: {cm[i, j]}<br>Percentage: {cm_percent[i, j]:.1f}%'
            )
        hover_text.append(hover_row)
    
    # Create annotations (count + percentage)
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f'{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)',
                    showarrow=False,
                    font=dict(
                        size=12,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black'
                    )
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        hovertext=hover_text,
        hoverinfo='text',
        colorbar=dict(title='Count')
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Confusion Matrix - {_extract_model_name(json_path)}',
            font=dict(size=16, family='Arial', color='#333')
        ),
        xaxis=dict(title='Predicted Label', side='bottom'),
        yaxis=dict(title='True Label', autorange='reversed'),
        template='plotly_white',
        height=500,
        width=600,
        annotations=annotations
    )
    
    if output_path:
        if output_path.endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_image(output_path, width=600, height=500)
        print(f"Saved confusion matrix to {output_path}")
    
    return fig


def _extract_model_name(json_path: str) -> str:
    """Extract readable model name from JSON filename."""
    path = Path(json_path)
    stem = path.stem
    
    if 'tier1' in stem.lower() and 'multi' not in stem.lower():
        return 'Tier1 Model'
    elif 'multicountry' in stem.lower() or 'multi_country' in stem.lower():
        return 'Multi-Country Model'
    else:
        return stem.replace('_', ' ').title()
