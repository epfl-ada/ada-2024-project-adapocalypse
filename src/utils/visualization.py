import plotly.express as px

def bar_plot_unique(df, xvalues, ylabel, title, labels):
    fig = px.bar(df, x=xvalues, title=title, labels=labels)
    fig.update_layout(yaxis_title=ylabel)
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()

def bar_graph(df, xvalues, yvalues, title, labels):
    fig = px.bar(df, x=xvalues, y=yvalues, title=title, labels=labels)
    fig.update_layout(
        autosize=True,
        xaxis=dict(tickangle=30),
    )
    fig.show()
    
def hist_plot_unique(df, xvalues, ylabel, title, nbins, labels):
    fig = px.histogram(df, x=xvalues, title=title, nbins=nbins, labels=labels)
    fig.update_layout(
        yaxis_title=ylabel
    )
    fig.show()