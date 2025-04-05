
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import transformer_lens.utils as utils
import pandas as pd

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        # color_continuous_scale="RdBu",
        # color_continuous_scale=[[0, "blue"], [1, "red"]],  # Explicitly blue to red
        color_continuous_scale="RdBu_r",  # reversed scale
        **kwargs,
    ).update_layout(coloraxis_colorbar=dict(x=.6)).update_layout(title_x=0.5, title_xanchor='center').show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()


def lines(tensors, names=None, x=None, xlabel='Layer',ylabel='IDim', **kwargs):
    # Convert each tensor to a NumPy array.
    arrays = [utils.to_numpy(tensor) for tensor in tensors]

    # Ensure all tensors have the same length.
    lengths = [len(arr) for arr in arrays]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("All tensors must have the same length.")

    # Use user-defined x-values if provided; otherwise, default to indices.
    if x is None:
        x = list(range(lengths[0]))
    else:
        x = utils.to_numpy(x)
        if len(x) != lengths[0]:
            raise ValueError("The length of x must match the length of each tensor.")

    # Use provided names or generate default ones.
    if names is None:
        names = [f"Tensor {i+1}" for i in range(len(arrays))]
    elif len(names) != len(arrays):
        raise ValueError("The length of 'names' must match the number of tensors.")

    # Assume hover labels are always provided via 'hover_name' in kwargs.
    if 'hover_name' not in kwargs:
        raise ValueError("hover_name must be provided in kwargs.")
    hover_labels = kwargs.pop('hover_name')
    if len(hover_labels) != lengths[0]:
        raise ValueError("The length of hover_name must match the length of each tensor.")

    # Create a DataFrame with x-values and hover labels.
    df = pd.DataFrame({
        'Layer': x,
        'hover_name': hover_labels,
    })

    # Add each tensor as a column.
    for name, arr in zip(names, arrays):
        df[name] = arr

    # Convert the DataFrame to long format, preserving hover labels.
    df_long = df.melt(id_vars=['Layer', 'hover_name'], var_name='Prompt', value_name='IDim')

    # Plot the lines with the hover labels coming from 'hover_name'.
    fig = px.line(df_long, x=xlabel, y=ylabel, color='Prompt', hover_name='hover_name', **kwargs)
    #fig = px.line(df_long, x='Layer', y='IDIM', hover_name='hover_name', **kwargs)
    fig.update_layout(font_size=20)
    fig.show()









def create_correlation_plot(x_data, y_data, title="Scatter Plot with Correlation Line", 
                           x_label="X Values", y_label="Y Values"):
    """
    Create a scatter plot with correlation line using Plotly.
    
    Parameters:
    -----------
    x_data : array-like
        The x-axis data points
    y_data : array-like
        The y-axis data points
    title : str, optional
        The title of the plot
    x_label : str, optional
        The label for the x-axis
    y_label : str, optional
        The label for the y-axis
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Plotly figure object
    """
    # Convert inputs to numpy arrays
    x = np.array(x_data)
    y = np.array(y_data)
    
    # Calculate Pearson correlation coefficient
    corr_coef, p_value = pearsonr(x, y)
    
    # Calculate line of best fit
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.array([min(x), max(x)])
    line_y = slope * line_x + intercept
    
    # Create the figure with scatter plot
    fig = make_subplots()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=y, 
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.7),
            name='Data Points'
        )
    )
    
    # Add correlation line
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Correlation Line (r = {corr_coef:.3f})'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        annotations=[
            dict(
                x=0.95,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"Pearson r = {corr_coef:.3f}<br>p-value = {p_value:.3e}",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    return fig
