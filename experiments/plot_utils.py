import plotly.express as px
import transformer_lens.utils as utils
import pandas as pd

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        # color_continuous_scale="RdBu",
        # color_continuous_scale=[[0, "blue"], [1, "red"]],  # Explicitly blue to red
        color_continuous_scale="RdBu_r",  # reversed scale
        **kwargs,
    ).show()


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

def lines(tensors, names=None, x=None, **kwargs):
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
    df_long = df.melt(id_vars=['Layer', 'hover_name'], var_name='Model', value_name='Intrinsic Dimension')

    # Plot the lines with the hover labels coming from 'hover_name'.
    fig = px.line(df_long, x='Layer', y='Intrinsic Dimension', color='Model', hover_name='hover_name', **kwargs)
    fig.show()