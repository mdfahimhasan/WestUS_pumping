# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def make_line_plot(y_series, year, fontsize, xlabel, ylabel,
                   line_labels, colors=None, markers=None,
                   figsize=(10, 4), y_lim=None, legend_pos='upper left', legend='on',
                   savepath=None, no_xticks=False, suptitle=None):
    """
    Flexible line plot for multiple series.

    Parameters
    ----------
    y_series : list of arrays/lists
        Each element is a series of y-values (e.g., [y1, y2, y3, y4]).
    year : list/array
        x-axis values (same length as each y series).
    fontsize : int
        Font size for labels, ticks, legend, and suptitle.
    xlabel, ylabel : str
        Axis labels.
    line_labels : list of str
        Labels for each line.
    colors : list of str, optional
        Colors for each line (default = matplotlib auto colors).
    markers : list of str, optional
        Markers for each line (default = different symbols).
    figsize : tuple, optional
        Figure size. Default is (10, 4).
    y_lim : tuple, optional
        y-axis limits. Default is None.
    legend_pos : str, optional
        Legend position. Default is 'upper left'.
    legend : {'on', 'off'}, optional
        Whether to display the legend. Default is 'on'.
    savepath : str, optional
        If provided, figure will be saved at this location.
    no_xticks : bool, optional
        If True, hides x-axis tick labels. Default is False.
    suptitle : str, optional
        Optional figure-level title. Default is None.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # defaults
    if colors is None:
        colors = ["#000000",  # black
                  "#E24A33",  # orange-red
                  "#348ABD",  # blue
                  "#988ED5",  # purple
                  "#8EBA42",  # green
                  "#777777",  # gray
                  "#FBC15E",  # gold
                  "#FFB5B8"  # pink
                  ]

        # colors = plt.cm.tab10.colors

    if markers is None:
        markers = ['o', 's', 'D', '^', 'v', 'P', '*']  # cycle of markers

    # plot all series
    linestyles = ['-', '-', '-', '-.']

    for i, y in enumerate(y_series):
        ax.plot(year, y,
                label=line_labels[i],
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                markersize=4,
                linewidth=1)

    # axis labels & limits
    ax.set_xticks(year)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylim(y_lim)

    # legend
    if legend == 'on':
        ax.legend(loc=legend_pos, fontsize=fontsize - 4)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # remove bounding box
    sns.despine(offset=10, trim=True)

    # suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize + 2)

    # xticks & yticks
    ax.set_xticklabels(labels=year, rotation=45, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    if no_xticks:
        ax.set_xticklabels(labels=[])

    plt.subplots_adjust(bottom=0.25)

    # save figure
    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True, bbox_inches='tight')


def make_line_plot_v1(y1, y2, year, fontsize, xlabel, ylabel, line_label_1, line_label_2,
                      plot1_color='tab:blue', plot2_color='tab:green',
                      figsize=(10, 4), y_lim=None, legend_pos='upper left', legend='on',
                      savepath=None, no_xticks=False, suptitle=None):
    # line plot (annual mean mm/year)
    fig, ax = plt.subplots(figsize=figsize)

    # sns.set(style='ticks')  # or 'darkgrid', 'ticks'
    # ax.set_facecolor('#f5f5f5')  # soft light gray
    # fig.patch.set_facecolor('#f5f5f5')

    ax.plot(year, y1, label=line_label_1, color=plot1_color, marker='D', markersize=6, linewidth=1)
    ax.plot(year, y2, label=line_label_2, color=plot2_color, marker='o', markersize=6, linewidth=1)
    ax.set_xticks(year)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(y_lim)

    # legend
    if legend == 'on':
        ax.legend(loc=legend_pos, fontsize=fontsize)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # remove bounding box
    sns.despine(offset=10, trim=True)  # turning of bounding box around the plots

    # suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    # xticks
    ax.set_xticklabels(labels=year, rotation=45, fontsize=fontsize)
    if no_xticks:
        ax.set_xticklabels(labels=[])

    plt.subplots_adjust(bottom=0.25)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True, bbox_inches='tight')


def make_line_plot_with_error(y_series, year, fontsize, xlabel, ylabel,
                              line_labels, colors=None, markers=None,
                              low_CI=None, high_CI=None,  # <-- NEW
                              figsize=(10, 4), y_lim=None, legend_pos='upper left',
                              legend='on', savepath=None, no_xticks=False, suptitle=None,
                              alpha_band=0.2):
    """
    Flexible line plot for multiple series with optional confidence interval (error bands).

    Parameters
    ----------
    y_series : list of arrays/lists
        Each element is a series of y-values (e.g., [y1, y2, y3]).
    year : list/array
        x-axis values (same length as each y series).
    fontsize : int
        Font size for labels, ticks, legend, and suptitle.
    xlabel, ylabel : str
        Axis labels.
    line_labels : list of str
        Labels for each line.
    colors : list of str, optional
        Colors for each line (default = predefined color palette).
    markers : list of str, optional
        Markers for each line (default = ['o', 's', 'D', '^', 'v', 'P', '*']).
    low_CI, high_CI : list of arrays/lists, optional
        Lower and upper confidence bounds for each series.
        Each must be the same length and order as `y_series`.
        If None, error bars/bands are not plotted.
    figsize : tuple, optional
        Figure size. Default is (10, 4).
    y_lim : tuple, optional
        y-axis limits. Default is None.
    legend_pos : str, optional
        Legend position. Default is 'upper left'.
    legend : {'on', 'off'}, optional
        Whether to display the legend. Default is 'on'.
    savepath : str, optional
        If provided, figure will be saved at this location.
    no_xticks : bool, optional
        If True, hides x-axis tick labels. Default is False.
    suptitle : str, optional
        Optional figure-level title. Default is None.
    alpha_band : float, optional
        Transparency for CI shading. Default is 0.2.

    Returns
    -------
    None
        Displays the plot and optionally saves the figure.

    Notes
    -----
    - Error regions are shaded between `low_CI` and `high_CI` for each line.
    - Ensure `low_CI[i]`, `high_CI[i]`, and `y_series[i]` all have the same length.
    - Works seamlessly with your precomputed pixel/basin-level confidence intervals.

    Example
    -------
    >>> make_line_plot_with_error(
    ...     y_series=[mean_pred, actual],
    ...     low_CI=[low_band, None],
    ...     high_CI=[high_band, None],
    ...     year=np.arange(2000, 2023),
    ...     line_labels=['Predicted pumping', 'Observed pumping'],
    ...     xlabel='Year',
    ...     ylabel='Pumping (mm)',
    ...     fontsize=12,
    ...     colors=['#348ABD', '#E24A33']
    ... )
    """

    fig, ax = plt.subplots(figsize=figsize)

    # defaults
    if colors is None:
        colors = [
            "#000000",  # black
            "#2ca02c",  # green
            "#E24A33",  # red / orange-red
            "#348ABD",  # blue
            "#988ED5",  # purple
            "#777777",  # gray
            "#FBC15E",  # gold
            "#FFB5B8"  # pink
        ]

    if markers is None:
        markers = ['o', 's', 'D', '^', 'v', 'P', '*']

    linestyles = ['-', '-', '-', '-.']

    # plot all series
    for i, y in enumerate(y_series):
        color = colors[i % len(colors)]
        ax.plot(year, y,
                label=line_labels[i],
                color=color,
                marker=markers[i],
                linestyle=linestyles[i],
                markersize=4,
                linewidth=1)

        # add confidence band if provided
        if low_CI is not None and high_CI is not None:
            if low_CI[i] is not None and high_CI[i] is not None:
                ax.fill_between(year,
                                low_CI[i],
                                high_CI[i],
                                color=color,
                                alpha=alpha_band,
                                linewidth=0)

    # axis labels & limits
    year_values = year.values
    ax.set_xticks(year_values[::2])
    ax.set_xticklabels(labels=year_values[::2])
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylim(y_lim)

    # legend
    if legend == 'on':
        ax.legend(loc=legend_pos, fontsize=fontsize - 4)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # remove bounding box
    sns.despine(offset=10, trim=True)

    # suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize + 2)

    # xticks & yticks
    ax.tick_params(axis='x', labelsize=fontsize, rotation=45)
    ax.tick_params(axis='y', labelsize=fontsize)
    if no_xticks:
        ax.set_xticklabels(labels=[])

    plt.subplots_adjust(bottom=0.25)

    # save figure
    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True, bbox_inches='tight')

    plt.show()


def make_pumping_comparison_scatter_plot(df, x, y, hue,
                                         xlabel, ylabel, fontsize, lim,
                                         scientific_ticks=True, scilimits=(4, 4),
                                         figsize=(6, 4), savepath=None,
                                         legend='on', legend_font=10,
                                         inset_basins=None, inset_lim=None,
                                         inset_lim_gap=5):
    """
    Plots a basin-wise scatter plot comparing predicted vs. actual pumping, with an optional inset zoom.

    :param df: pd.DataFrame
        DataFrame containing the variables for plotting.
    :param x: str
        Column name representing the predicted pumping values (x-axis).
    :param y: str
        Column name representing the actual pumping values (y-axis).
    :param hue: str
        Column name used to group points by color (e.g., 'basin').
    :param xlabel: str
        Label for the x-axis.
    :param ylabel: str
        Label for the y-axis.
    :param fontsize: int
        Font size for labels and ticks.
    :param lim: tuple of (float, float)
        Axis limits for both x and y axes (assumes square plot).
    :param scientific_ticks: bool, optional (default=True)
        Whether to format ticks in scientific notation.
    :param scilimits: tuple, optional (default=(4, 4))
        Limits for triggering scientific notation in tick labels.
    :param figsize: tuple, optional (default=(6, 4))
        Size of the overall figure in inches.
    :param savepath: str or None, optional (default=None)
        File path to save the figure. If None, figure is not saved.
    :param legend: str, optional (default='on')
        If 'on', displays the legend; otherwise, legend is removed.
    :param legend_font: Legend's font size. Default set to 10.
    :param inset_basins: list of str or None, optional
        List of basins to include in an inset zoom plot. If None, inset is not shown.
    :param inset_lim: tuple or None, optional
        Axis limits for the inset plot (x and y). If None, defaults to (0, 30).
    :param inset_lim_gap: xlim and ylim intervals for the inset. Default set to 5.

    :return: None
    """

    basin_colors = {
        'GMD4, KS': '#4c72b0',
        'GMD3, KS': '#dd8452',
        'Republican River Basin, CO': '#55a868',
        'South Platte River Basin, CO': '#e377c2',
        'Arkansas River Basin, CO': '#ffb000',
        'Rio Grande River Basin, CO': '#17becf',
        'Harquahala INA, AZ': '#c44e52',
        'Douglas AMA, AZ': '#8172b3',
        'Phoenix AMA, AZ': '#937860',
        'Pinal AMA, AZ': '#8c8c8c',
        'Tucson AMA, AZ': '#EE6677',
        'Santa Cruz AMA, AZ': '#009E73',
        'Diamond Valley, NV': '#E69F00',
        'Parowan Valley, UT': '#E69F00'
    }

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['font.size'] = fontsize

    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=basin_colors,
                    marker='s', s=30, ax=ax)

    ax.plot([0, 1], [0, 1], 'gray', transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    if scientific_ticks:
        ax.ticklabel_format(style='sci', scilimits=scilimits)
        ax.tick_params(axis='both', labelsize=fontsize)

    if legend != 'on':
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                  loc='upper left',
                  fontsize=legend_font,
                  markerscale=0.8,
                  frameon=True,
                  handlelength=1,
                  handletextpad=0.2,
                  labelspacing=0.3,
                  borderaxespad=0.3)

    # Optional inset
    if inset_basins:
        inset_df = df[df[hue].isin(inset_basins)]
        inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower right',
                              bbox_to_anchor=(0.2, 0.1, 0.8, 0.8),  # adjust here for inset position and size
                              bbox_transform=ax.transAxes, borderpad=0)

        sns.scatterplot(data=inset_df, x=x, y=y, hue=hue, palette=basin_colors,
                        marker='s', s=30, ax=inset_ax, legend=False)
        inset_ax.plot([0, 1], [0, 1], 'gray', transform=inset_ax.transAxes)
        inset_ax.set_xlabel('')
        inset_ax.set_ylabel('')
        inset_ax.set_xlim(inset_lim if inset_lim else (0, 30))
        inset_ax.set_ylim(inset_lim if inset_lim else (0, 30))
        inset_ax.xaxis.set_major_locator(MultipleLocator(inset_lim_gap))
        inset_ax.yaxis.set_major_locator(MultipleLocator(inset_lim_gap))
        inset_ax.tick_params(labelsize=fontsize - 4)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=400, transparent=True)


def join_plot_together(n_cols, plots_to_join, output_plot,
                       fig_size=(7, 8), titles=None, title_fontsize=10):
    n_rows = (len(plots_to_join) + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axs = axs.flatten()

    for i, path in enumerate(plots_to_join):
        img = mpimg.imread(path)
        axs[i].imshow(img)
        axs[i].axis('off')

        if titles is not None and i < len(titles):
            axs[i].set_title(titles[i], fontsize=title_fontsize)

    # hiding unused axes if any
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # saving plot
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_plot, dpi=400, bbox_inches='tight')


def plot_heatmap(df, title, cbar_label,
                 fontsize=12, cmap='coolwarm',
                 annot=True, figsize=(7, 5),
                 fmt='.2f', mask_diag=True,
                 savepath=None):
    """
    Plots a heatmap for pairwise statistics (for example, RMSE, R2, P-value).

    Parameters
    ----------
    df : pd.DataFrame
        Square DataFrame containing pairwise statistics.
    title : str
        Title of the heatmap.
    cbar_label: str
        Colorbar label.
    fontsize: int
        Fontsize.
    cmap : str, default='coolwarm'
        Colormap to use for the heatmap.
    annot : bool, default=True
        Whether to annotate the cells with values.
    figsize : tuple, default=(7, 5)
        Figure size in inches.
    fmt : str, default='.2f'
        String formatting code for annotations.
    mask_diag : bool, default=True
        Whether to hide (mask) the diagonal cells.
    savepath : str or None, optional
        If provided, saves the heatmap to this file path.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The heatmap axes object.
    """

    # Optional mask for diagonal
    mask = None
    if mask_diag:
        mask = np.eye(len(df), dtype=bool)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap,
                     linewidths=0.5, mask=mask,
                     cbar_kws={'label': cbar_label},
                     vmin=0, vmax=1)

    if title is not None:
        ax.set_title(title, fontsize=fontsize, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=90, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)

    if annot:
        for text in ax.texts:
            text.set_fontsize(fontsize)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
