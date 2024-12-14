import plotly.express as px
import matplotlib.pyplot as plt

def movies_by_genre(df, xvalues, yvalues, title, labels):
    fig = px.bar(df, x=xvalues, y=yvalues, title=title, labels=labels)
    fig.update_traces(textfont_size=10, textangle=30, textposition="outside", cliponaxis=False)
    fig.show()

def movies_by_country(df, xvalues, yvalues, title, labels):
    fig = px.bar(df, x=xvalues, y=yvalues, title=title, labels=labels)
    fig.update_layout(
        autosize=True,
        xaxis=dict(tickangle=30),
    )
    fig.show()
    
def movies_per_year(df, xvalues, ylabel, title, nbins, labels):
    fig = px.histogram(df, x=xvalues, title=title, nbins=nbins, labels=labels)
    fig.update_layout(
        yaxis_title=ylabel
    )
    fig.show()
    
def nb_char_per_movie(df, xvalues, ylabel, title, nbins, labels):
    fig = px.histogram(df, x=xvalues, title=title, nbins=nbins, labels=labels)
    fig.update_layout(
        yaxis_title=ylabel
    )
    fig.show()
    
def char_by_dir_gender(df, xvalues, ylabel, title, color, labels):
    fig = px.bar(df, x=xvalues, title=title, labels=labels, color=color)
    fig.update_layout(
        yaxis_title=ylabel
    )
    fig.show()
    
    
def nb_char_through_years(df, xvalues, title, labels):
    fig, ax = plt.subplots(figsize=(15,6))
    width = 0.4
    x = df.index  # Years
    x_male = x - width / 2
    x_female = x + width / 2

    # Plot bars side by side
    ax.bar(x_male, df[0], width, label="Male", color="gold")
    ax.bar(x_female, df[1], width, label="Female", color="royalblue")
    ax.set_title("Evolution of number of characters")
    ax.legend()
    ax.set_ylabel("Number of characters")
    ax.set_xlabel("Years")
    ax.set_xticks(x[::10])  # Adjust to show ticks every 30 years
    ax.set_xticklabels(df[xvalues][::10]);
    
def evolution_fem_char(df, xvalues, yvalues, yerr, title):
    y_err = df[yerr]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df[xvalues], df[yvalues], "-o", label='Count', color="royalblue", markersize=4)
    ax.fill_between(df[xvalues], df[yvalues] - y_err, df[yvalues] + y_err, alpha=0.2, label='±1 Std. Dev.')
    ax.set_title(title)
    ax.set_xlabel("Years")
    ax.set_ylabel("Number of characters")
    ax.legend()
    
def distribution_actor_age(df, xvalues, male_label, women_label, male_avg, female_avg, title):
    fig, ax = plt.subplots()

    ax.bar(df[xvalues], df[0], label=male_label, color="gold")
    ax.bar(df[xvalues], df[1], label=women_label, color="royalblue")

    ax.axvline(male_avg, color="red", linestyle="--", linewidth=2, label=f"Avg Male Age: {male_avg:.0f}")
    ax.axvline(female_avg, color="red", linestyle="-", linewidth=2, label=f"Avg Female Age: {female_avg:.0f}")

    #get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # #specify order of items in legend
    order = [2, 3, 0, 1]

    #add legend to plot
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax.set_title(title)
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of actors");