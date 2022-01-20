import plotly.graph_objects as go
from IPython import display


def plot_history(history, name_history='loss', title=None, is_clear_before_draw=False):
    if is_clear_before_draw:
        display.clear_output(wait=True)

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(y=history['train'], x=list(range(len(history['train']))),
                             name=f'train {name_history}'))

    fig.add_trace(go.Scatter(y=history['val'], x=list(range(len(history['val']))),
                             name=f'val {name_history}'))

    # Edit the layout
    fig.update_layout(
        title=f'Plot {title if title is not None else name_history} during epochs',
        xaxis_title='epoch',
        yaxis_title=f'{name_history}',
        autosize=False,
        width=800,
        height=500,
    )

    fig.show()
