from abc import ABC
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple  # Classes
from typing import Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import axes
from matplotlib import gridspec
from matplotlib import pyplot as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure  # SubplotParams as sppars,

_VarNames = Union[str, Sequence[str]]
ColorLike = Union[str, Tuple[float, ...]]


class _AxesSubplot(Axes, axes.SubplotBase, ABC):
    """Intersection between Axes and SubplotBase: Has methods of both"""


# -------------------------------------------------------------------------------
# Simple plotting functions
# -------------------------------------------------------------------------------
# We build the MaskedDotPlot based on sc.pl.DotPlot. Compared to scanpy's dotplot
# the only add-ons are identifiers around dots e.g. for tree genes and marker genes.
# Unfortunately a lot of class methods needed to be copied and sligthly modified.
# Maybe this could be done more efficiently. Also, the legend add-ons for the identifiers
# have some positioning problems which need to be fixed.


# def matrix(
#    matrix,
#    xlabel=None,
#    ylabel=None,
#    xticks=None,
#    yticks=None,
#    title=None,
#    colorbar_shrink=0.5,
#    color_map=None,
#    show=None,
#    save=None,
#    ax=None,
# ):
#    """Plot a matrix."""
#    if ax is None:
#        ax = pl.gca()
#    img = ax.imshow(matrix, cmap=color_map)
#    if xlabel is not None:
#        ax.set_xlabel(xlabel)
#    if ylabel is not None:
#        ax.set_ylabel(ylabel)
#    if title is not None:
#        ax.set_title(title)
#    if xticks is not None:
#        ax.set_xticks(range(len(xticks)), xticks, rotation="vertical")
#    if yticks is not None:
#        ax.set_yticks(range(len(yticks)), yticks)
#    pl.colorbar(img, shrink=colorbar_shrink, ax=ax)  # need a figure instance for colorbar
#    savefig_or_show("matrix", show=show, save=save)


def fix_kwds(kwds_dict: dict, **kwargs):
    """Merge parameters from dictionary and kwargs.

    Note:
        Given a dictionary of plot parameters (kwds_dict) and a dict of kwds,
        merge the parameters into a single consolidated dictionary to avoid
        argument duplication errors.
        If kwds_dict an kwargs have the same key, only the value in kwds_dict is kept.

    Args:
        kwds_dict:
            kwds_dictionary
        kwargs

    Returns:
        kwds_dict merged with kwargs

    Example:
        .. code-block:: python

            def _example(**kwds):
                return fix_kwds(kwds, key1="value1", key2="value2")
            example(key1="value10", key3="value3")

        ::

            {'key1': 'value10, 'key2': 'value2', 'key3': 'value3'}

    """

    kwargs.update(kwds_dict)

    return kwargs


def make_grid_spec(
    ax_or_figsize: Union[Tuple[int, int], _AxesSubplot],
    nrows: int,
    ncols: int,
    wspace: Optional[float] = None,
    hspace: Optional[float] = None,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
) -> Tuple[Figure, gridspec.GridSpecBase]:
    kw = dict(
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    if isinstance(ax_or_figsize, tuple):
        fig = pl.figure(figsize=ax_or_figsize)
        return fig, gridspec.GridSpec(nrows, ncols, **kw)
    else:
        ax = ax_or_figsize
        ax.axis("off")
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)


def _get_basis(adata: anndata.AnnData, basis: str):

    if basis in adata.obsm.keys():
        basis_key = basis

    elif f"X_{basis}" in adata.obsm.keys():
        basis_key = f"X_{basis}"

    return basis_key


class MaskedDotPlot(sc.pl.DotPlot):
    """Dotplot with additional annotation masks.

    Args:
        adata:
            AnnData with ``adata.obs[groupby]`` cell type annotations.
        var_names:
            ``var_names`` should be a valid subset of ``adata.var_names``. If ``var_names`` is a mapping, then the key
            is used as label to group the values (see ``var_group_labels``). The mapping values should be sequences of
            valid ``adata.var_names``. In this case either coloring or ‘brackets’ are used for the grouping of var names
            depending on the plot. When ``var_names`` is a mapping, then the ``var_group_labels`` and
            ``var_group_positions`` are set.
        groupby:
            The key of the observation grouping to consider.
        tree_genes:
            Dictionary with lists of forest selected genes.
        marker_genes:
            Dictionary with list of marker genes for each celltype.
        further_celltypes:
            Celltypes that are not in ``adata.obs[groupby]``.
        tree_genes_color:
            Color for ``tree_genes``.
        marker_genes_color:
            Color for ``marker_genes``.
        non_adata_celltypes_color:
            Color for celltypes that don't occur in ``adata``.
        use_raw:
            Use ``raw`` attribute of ``adata`` if present.
        log:
            Plot on logarithmic axis.
        num_categories:
            Only used if groupby observation is not categorical. This value determines the number of groups into which
            the groupby observation should be subdivided.
        categories_order:
            Order in which to show the categories. Note: add_dendrogram or add_totals can change the categories order.
        title:
            Title for the figure
        figsize:
            Figure size when ``multi_panel=True``. Otherwise the ``rcParam['figure.figsize]`` value is used. Format is
            `(width, height)`.
        gene_symbols:
            Column name in ``adata.var`` DataFrame that stores gene symbols. By default, ``var_names`` refer to the
            index
            column of the ``adata.var`` DataFrame. Setting this option allows alternative names to be used.
        var_group_positions:
            Use this parameter to highlight groups of ``var_names``. This will draw a ‘bracket’ or a color block between
            the given start and end positions. If the parameter ``var_group_labels`` is set, the corresponding labels
            are added on top/left. E.g. ``var_group_positions=[(4,10)]`` will add a bracket between the fourth
            ``var_name`` and the tenth ``var_name``. By giving more positions, more brackets/color blocks are drawn.
        var_group_labels:
            Labels for each of the ``var_group_positions`` that want to be highlighted.
        var_group_rotation:
            Label rotation degrees. By default, labels larger than 4 characters are rotated 90 degrees.
        layer:
            Name of the AnnData object layer that wants to be plotted. By default, ``adata.raw.X`` is plotted. If
            ``use_raw=False`` is set, then ``adata.X`` is plotted. If ``layer`` is set to a valid layer name, then the
            layer is plotted. ``layer`` takes precedence over ``use_raw``.
        expression_cutoff:
            Expression cutoff that is used for binarizing the gene expression and determining the fraction of cells
            expressing given genes. A gene is expressed only if the expression value is greater than this threshold.
        mean_only_expressed:
            If `True`, gene expression is averaged only over the cells expressing the given genes.
        standard_scale:
            Whether or not to standardize that dimension between 0 and 1, meaning for each variable or group, subtract
            the minimum and divide each by its maximum.
        dot_color_df:
            Data frame containing the dot size.
        dot_size_df:
            Data frame containing the dot color, should have the same, shape, columns and indices as ``dot_size``.
        ax:
            Matplotlib axes.
        grid:
            Adds a grid to the plot.
        grid_linewidth:
            Matplotlib linewidth.
        **kwds:
            Are passed to :func:`matplotlib.pyplot.scatter`.
"""

    # TODO proofread docstring

    def __init__(
        self,
        adata: AnnData,
        var_names: Union[_VarNames, Mapping[str, _VarNames]],
        groupby: Union[str, Sequence[str]] = "celltype",
        tree_genes: Optional[dict] = None,
        marker_genes: Optional[dict] = None,
        further_celltypes: Optional[Sequence[str]] = None,
        tree_genes_color: Optional[str] = "darkblue",
        marker_genes_color: Optional[str] = "green",
        non_adata_celltypes_color: Optional[str] = "grey",
        use_raw: Optional[bool] = None,
        log: bool = False,
        num_categories: int = 7,
        categories_order: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        gene_symbols: Optional[str] = None,
        var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
        var_group_labels: Optional[Sequence[str]] = None,
        var_group_rotation: Optional[float] = None,
        layer: Optional[str] = None,
        expression_cutoff: float = 0.0,
        mean_only_expressed: bool = False,
        standard_scale: Literal["var", "group"] = "var",
        dot_color_df: Optional[pd.DataFrame] = None,
        dot_size_df: Optional[pd.DataFrame] = None,
        ax: Optional[_AxesSubplot] = None,
        grid: Optional[bool] = True,
        grid_linewidth: Optional[float] = 0.1,
        **kwds,
    ):
        self.tree_genes = tree_genes
        self.marker_genes = marker_genes
        self.further_celltypes = further_celltypes
        self.show_marker_legend = True
        self.tree_genes_color = tree_genes_color
        self.marker_genes_color = marker_genes_color
        self.non_adata_celltypes_color = non_adata_celltypes_color
        self.grid_linewidth = grid_linewidth
        sc.pl.DotPlot.__init__(
            self,
            adata,
            var_names,
            groupby,
            use_raw=use_raw,
            log=log,
            num_categories=num_categories,
            categories_order=categories_order,
            title=title,
            figsize=figsize,
            gene_symbols=gene_symbols,
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=var_group_rotation,
            layer=layer,
            ax=ax,
            expression_cutoff=expression_cutoff,
            mean_only_expressed=mean_only_expressed,
            standard_scale=standard_scale,
            dot_color_df=dot_color_df,
            dot_size_df=dot_size_df,
            **kwds,
        )
        self.style(
            cmap="Reds",
            color_on="dot",
            # 'square',
            # dot_max=None,
            # dot_min=None,
            # smallest_dot=0.0,
            # largest_dot=200.0,
            # dot_edge_color='black',
            # dot_edge_lw=0.2,
            # size_exponent=1.5,
            grid=grid,  # True,
            # x_padding=0.8,
            # y_padding=1.0)
        )
        # default height and width are 0.35 and 0.37. Since we use symmetrical markers we set them to the same values
        # (this restricts the flexibility of the function... we'd like to keep the generality of sc.pl.dotplot, maybe
        # this can be done better)
        default_length = (self.DEFAULT_CATEGORY_WIDTH + self.DEFAULT_CATEGORY_HEIGHT) / 2  # type: ignore
        self.DEFAULT_CATEGORY_WIDTH = default_length
        self.DEFAULT_CATEGORY_HEIGHT = default_length

        if further_celltypes:
            self.DEFAULT_CATEGORY_HEIGHT = self.DEFAULT_CATEGORY_HEIGHT * (
                (len(further_celltypes) + len(self.categories)) / (len(self.categories))
            )

    def _mainplot(self, ax):
        # work on a copy of the dataframes. This is to avoid changes
        # on the original data frames after repetitive calls to the
        # DotPlot object, for example once with swap_axes and other without

        _color_df = self.dot_color_df.copy()
        _size_df = self.dot_size_df.copy()

        if self.var_names_idx_order is not None:
            _color_df = _color_df.iloc[:, self.var_names_idx_order]
            _size_df = _size_df.iloc[:, self.var_names_idx_order]

        if self.categories_order is not None:
            _color_df = _color_df.loc[self.categories_order, :]
            _size_df = _size_df.loc[self.categories_order, :]

        if self.further_celltypes:
            _other_celltypes_df = pd.DataFrame(index=_color_df.index, columns=_color_df.columns, data=False)
            for ct in self.further_celltypes:
                _color_df.loc[ct] = 0
                _size_df.loc[ct] = 0
                _other_celltypes_df.loc[ct] = True
        else:
            _other_celltypes_df = None
        if self.tree_genes:
            _trees_df = pd.DataFrame(index=_color_df.index, columns=_color_df.columns, data=False)
            for ct, genes in self.tree_genes.items():
                _trees_df.loc[ct, genes] = True
        else:
            _trees_df = None
        if self.marker_genes:
            _marker_df = pd.DataFrame(index=_color_df.index, columns=_color_df.columns, data=False)
            for ct, genes in self.marker_genes.items():
                _marker_df.loc[ct, genes] = True
        else:
            _marker_df = None

        if self.are_axes_swapped:
            _size_df = _size_df.T
            _color_df = _color_df.T
            if _other_celltypes_df:
                _other_celltypes_df = _other_celltypes_df.T
            if _trees_df:
                _trees_df = _trees_df.T
            if _marker_df:
                _marker_df = _marker_df.T
        self.cmap = self.kwds.get("cmap", self.cmap)
        if "cmap" in self.kwds:
            del self.kwds["cmap"]

        normalize, dot_min, dot_max = self._dotplot(
            _size_df,
            _color_df,
            ax,
            _trees_df,
            _marker_df,
            _other_celltypes_df,
            tree_genes_color=self.tree_genes_color,
            marker_genes_color=self.marker_genes_color,
            non_adata_celltypes_color=self.non_adata_celltypes_color,
            cmap=self.cmap,
            dot_max=self.dot_max,
            dot_min=self.dot_min,
            color_on=self.color_on,
            edge_color=self.dot_edge_color,
            edge_lw=self.dot_edge_lw,
            smallest_dot=self.smallest_dot,
            largest_dot=self.largest_dot,
            size_exponent=self.size_exponent,
            grid=self.grid,
            grid_linewidth=self.grid_linewidth,
            x_padding=self.plot_x_padding,
            y_padding=self.plot_y_padding,
            **self.kwds,
        )

        self.dot_min, self.dot_max = dot_min, dot_max
        return normalize

    @staticmethod
    def _dotplot(
        dot_size,
        dot_color,
        dot_ax,
        tree_genes,
        marker_genes,
        grey_celltypes,
        tree_genes_color="darkblue",
        marker_genes_color="green",
        non_adata_celltypes_color="grey",
        cmap: str = "Reds",
        color_on: Optional[str] = "dot",
        y_label: Optional[str] = None,
        dot_max: Optional[float] = None,
        dot_min: Optional[float] = None,
        standard_scale: Literal["var", "group"] = None,
        smallest_dot: Optional[float] = 0.0,
        largest_dot: Optional[float] = 200,
        size_exponent: Optional[float] = 2,
        edge_color: Optional[ColorLike] = None,
        edge_lw: Optional[float] = None,
        grid: Optional[bool] = False,
        grid_linewidth: Optional[float] = 0.1,
        x_padding: Optional[float] = 0.8,
        y_padding: Optional[float] = 1.0,
        **kwds,
    ):
        """\
        Makes a *dot plot* given two data frames, one containing
        the doc size and other containing the dot color. The indices and
        columns of the data frame are used to label the output image
        The dots are plotted using :func:`matplotlib.pyplot.scatter`. Thus, additional
        arguments can be passed.
        Args:
            dot_size:
                Data frame containing the dot size.
            dot_color:
                Data frame containing the dot color, should have the same, shape, columns and indices as ``dot_size``.
            dot_ax:
                Matplotlib axis.
            tree_genes_color
                Color for tree genes.
            marker_genes_color
                Color for marker genes.
            non_adata_celltypes_color
                Color for celltypes that don't occur in ``adata``.
            cmap
                String denoting matplotlib color map.
            color_on
                Options are 'dot' or 'square'. By default the colomap is applied to the color of the dot. Optionally,
                the colormap can be applied to a square behind the dot, in which case the dot is transparent and only
                the edge is shown.
            y_label:
                Label for y-axis.
            dot_max
                If `None`, the maximum dot size is set to the maximum fraction value found (e.g. 0.6). If given, the
                value should be a number between 0 and 1. All fractions larger than ``dot_max`` are clipped to this
                value.
            dot_min
                If `None`, the minimum dot size is set to 0. If given, the value should be a number between 0 and 1. All
                fractions smaller than ``dot_min`` are clipped to this value.
            standard_scale
                Whether or not to standardize that dimension between 0 and 1, meaning for each variable or group,
                subtract the minimum and divide each by its maximum.
            smallest_dot
                If none, the smallest dot has size 0. All expression levels with ``dot_min`` are plotted with this size.
            edge_color
                Dot edge color. When ``color_on='dot'`` the default is no edge. When ``color_on='square'``, edge color
                is white.
            edge_lw
                Dot edge line width. When ``color_on='dot'`` the default is no edge. When ``color_on='square'``, line
                width = 1.5
            grid
                Adds a grid to the plot.
            x_paddding
                Space between the plot left/right borders and the dots center. A unit is the distance between the x
                ticks. Only applied when ``color_on = 'dot'``.
            y_paddding
                Space between the plot top/bottom borders and the dots center. A unit is the distance between the y
                ticks. Only applied when ``color_on = 'dot'``.
            kwds
                Are passed to :func:`matplotlib.pyplot.scatter`.

        Returns:
            matplotlib.colors.Normalize, dot_min, dot_max
        """

        assert dot_size.shape == dot_color.shape, (
            "please check that dot_size " "and dot_color dataframes have the same shape"
        )

        assert list(dot_size.index) == list(dot_color.index), (
            "please check that dot_size " "and dot_color dataframes have the same index"
        )

        assert list(dot_size.columns) == list(dot_color.columns), (
            "please check that the dot_size " "and dot_color dataframes have the same columns"
        )

        if standard_scale == "group":
            dot_color = dot_color.sub(dot_color.min(1), axis=0)
            dot_color = dot_color.div(dot_color.max(1), axis=0).fillna(0)
        elif standard_scale == "var":
            dot_color -= dot_color.min(0)
            dot_color = (dot_color / dot_color.max(0)).fillna(0)
        elif standard_scale is None:
            pass

        # make scatter plot in which
        # x = var_names
        # y = groupby category
        # size = fraction
        # color = mean expression

        # +0.5 in y and x to set the dot center at 0.5 multiples
        # this facilitates dendrogram and totals alignment for
        # matrixplot, dotplot and stackec_violin using the same coordinates.
        y, x = np.indices(dot_color.shape)
        y = y.flatten() + 0.5
        x = x.flatten() + 0.5
        frac = dot_size.values.flatten()
        mean_flat = dot_color.values.flatten()
        if isinstance(grey_celltypes, pd.DataFrame):
            grey_celltypes_mask = grey_celltypes.values.flatten()
        if isinstance(tree_genes, pd.DataFrame):
            tree_genes_mask = tree_genes.values.flatten()
        if isinstance(marker_genes, pd.DataFrame):
            marker_genes_mask = marker_genes.values.flatten()
        cmap = pl.get_cmap(kwds.get("cmap", cmap))
        if "cmap" in kwds:
            del kwds["cmap"]
        if dot_max is None:
            dot_max = np.ceil(max(frac) * 10) / 10
        else:
            if dot_max < 0 or dot_max > 1:
                raise ValueError("`dot_max` value has to be between 0 and 1")
        if dot_min is None:
            dot_min = 0
        else:
            if dot_min < 0 or dot_min > 1:
                raise ValueError("`dot_min` value has to be between 0 and 1")

        if dot_min != 0 or dot_max != 1:
            # clip frac between dot_min and  dot_max
            frac = np.clip(frac, dot_min, dot_max)
            old_range = dot_max - dot_min
            # re-scale frac between 0 and 1
            frac = (frac - dot_min) / old_range

        size = frac ** size_exponent
        # rescale size to match smallest_dot and largest_dot
        size = size * (largest_dot - smallest_dot) + smallest_dot  # type: ignore

        import matplotlib.colors

        normalize = matplotlib.colors.Normalize(vmin=kwds.get("vmin"), vmax=kwds.get("vmax"))

        if color_on == "square":
            if edge_color is None:
                from seaborn.utils import relative_luminance

                # use either black or white for the edge color
                # depending on the luminance of the background
                # square color
                edge_color = []  # type: ignore
                for color_value in cmap(normalize(mean_flat)):  # type: ignore
                    lum = relative_luminance(color_value)
                    edge_color.append(".15" if lum > 0.408 else "w")  # type: ignore

            edge_lw = 1.5 if edge_lw is None else edge_lw

            # first make a heatmap similar to `sc.pl.matrixplot`
            # (squares with the asigned colormap). Circles will be plotted
            # on top
            dot_ax.pcolor(dot_color.values, cmap=cmap, norm=normalize)
            for axis in ["top", "bottom", "left", "right"]:
                dot_ax.spines[axis].set_linewidth(1.5)
            kwds = fix_kwds(
                kwds,
                s=size,
                cmap=cmap,
                norm=None,
                linewidth=edge_lw,
                facecolor="none",
                edgecolor=edge_color,
            )
            dot_ax.scatter(x, y, **kwds)
        else:
            edge_color = "none" if edge_color is None else edge_color
            edge_lw = 0.0 if edge_lw is None else edge_lw

            # TODO: i think the factor 1.5 can be infered from some variable, but i don't know which...
            # In case you want to make the edges nicer consider also plotting white overlapping squares on all genes
            # that are not tree genes so that the outer edge half is not plotted (white square is overlapping). The
            # correct order of the (then) four scatter plots is important
            if isinstance(tree_genes, pd.DataFrame):
                dot_ax.scatter(
                    x[tree_genes_mask],
                    y[tree_genes_mask],
                    s=np.max(size) * 1.55,
                    marker="s",
                    color="none",
                    linewidth=2,
                    edgecolor=tree_genes_color,
                )
                # dot_ax.scatter(x[~tree_genes_mask], y[~tree_genes_mask], s=np.max(size)*1.85, marker='s',
                # color='white',edgecolor='none')
                # Tried it, but without the second line and 1.55 instead of 1.85 just looks better
            if isinstance(grey_celltypes, pd.DataFrame):
                dot_ax.scatter(
                    x[grey_celltypes_mask],
                    y[grey_celltypes_mask],
                    s=np.max(size) * 1.95,
                    marker="s",
                    color=non_adata_celltypes_color,
                    edgecolor="none",
                    alpha=0.3,
                )
            if isinstance(marker_genes, pd.DataFrame):
                dot_ax.scatter(
                    x[marker_genes_mask],
                    y[marker_genes_mask],
                    s=np.max(size) * 1.85,
                    marker="s",
                    color=marker_genes_color,
                    edgecolor="none",
                    alpha=0.3,
                )

            color = cmap(normalize(mean_flat))  # type: ignore
            kwds = fix_kwds(
                kwds,
                s=size,
                cmap=cmap,
                color=color,
                norm=None,
                linewidth=edge_lw,
                edgecolor=edge_color,
            )

            dot_ax.scatter(x, y, **kwds)

        y_ticks = np.arange(dot_color.shape[0]) + 0.5
        dot_ax.set_yticks(y_ticks)
        dot_ax.set_yticklabels([dot_color.index[idx] for idx, _ in enumerate(y_ticks)], minor=False)

        x_ticks = np.arange(dot_color.shape[1]) + 0.5
        dot_ax.set_xticks(x_ticks)
        dot_ax.set_xticklabels(
            [dot_color.columns[idx] for idx, _ in enumerate(x_ticks)],
            rotation=90,
            ha="center",
            minor=False,
        )
        dot_ax.tick_params(axis="both", labelsize="small")
        dot_ax.grid(False)
        dot_ax.set_ylabel(y_label)

        # to be consistent with the heatmap plot, is better to
        # invert the order of the y-axis, such that the first group is on
        # top
        dot_ax.set_ylim(dot_color.shape[0], 0)
        dot_ax.set_xlim(0, dot_color.shape[1])

        if color_on == "dot":
            # add padding to the x and y lims when the color is not in the square
            # default y range goes from 0.5 to num cols + 0.5
            # and default x range goes from 0.5 to num rows + 0.5, thus
            # the padding needs to be corrected.
            x_padding = x_padding - 0.5  # type: ignore
            y_padding = y_padding - 0.5  # type: ignore
            dot_ax.set_ylim(dot_color.shape[0] + y_padding, -y_padding)

            dot_ax.set_xlim(-x_padding, dot_color.shape[1] + x_padding)

        if grid:
            dot_ax.grid(True, color="gray", linewidth=grid_linewidth)  # linewidth=0.1)
            dot_ax.set_axisbelow(True)

        return normalize, dot_min, dot_max

    def _plot_marker_legend(
        self,
        marker_legend_ax: Axes,
        tree_genes_color="darkblue",
        marker_genes_color="green",
        non_adata_celltypes_color="grey",
        i=0,
    ):
        """New"""
        size = self.dot_max * 1.55 * 150

        ax = marker_legend_ax
        if i == 0:
            ax.scatter([0], [0], s=size, marker="s", color="none", linewidth=2, edgecolor=tree_genes_color)
            ax.set_title("Spapros marker", size="small")
        elif i == 1:
            ax.scatter([0], [0], s=size, marker="s", color=marker_genes_color, edgecolor="none", alpha=0.3)
            ax.set_title("DE or lit. gene", size="small")
        elif i == 2:
            ax.scatter([0], [0], s=size, marker="s", color=non_adata_celltypes_color, edgecolor="none", alpha=0.3)
            ax.set_title("celltype not in dataset", size="small")

        # ymax = ax.get_ylim()[1]
        ax.set_ylim(-10, 10)
        # size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size='small')

        # ax.set_ylim([-1.5,1.5])
        # ax.set_yticks([1,0,-1])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.yaxis.set_label_position("right")
        # ax.tick_params(labelsize='small',length=0) # title fontsize
        # ax.yaxis.tick_right()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)

    def _plot_legend(self, legend_ax, return_ax_dict, normalize):

        # to maintain the fixed height size of the legends, a
        # spacer of variable height is added at the bottom.
        # The structure for the legends is:
        # first row: variable space to keep the other rows of
        #            the same size (avoid stretching)
        # second row: legend for dot size
        # third row: spacer to avoid color and size legend titles to overlap
        # fourth row: colorbar

        cbar_legend_height = self.min_figure_height * 0.08
        size_legend_height = self.min_figure_height * 0.27
        marker_legend_height = self.min_figure_height * 0.17  # 0.20
        spacer_height = self.min_figure_height * 0.3
        small_spacer_height = self.min_figure_height * 0.12  # 0.15

        height_ratios = [
            # self.height - size_legend_height - cbar_legend_height - spacer_height,
            self.height
            - 3 * marker_legend_height
            - size_legend_height
            - cbar_legend_height
            - spacer_height  # - 2 * spacer_height
            - 3 * small_spacer_height,  # - 2 * small_spacer_height,
            marker_legend_height,
            small_spacer_height,
            marker_legend_height,
            small_spacer_height,
            marker_legend_height,
            small_spacer_height,  # spacer_height,
            size_legend_height,
            spacer_height,
            cbar_legend_height,
        ]
        fig, legend_gs = make_grid_spec(legend_ax, nrows=10, ncols=1, height_ratios=height_ratios)

        ### Added
        if self.show_marker_legend:
            for i in range(3):
                marker_legend_ax = fig.add_subplot(legend_gs[i * 2 + 1])  # legend_gs[i + 1 + int(i > 0)])
                self._plot_marker_legend(
                    marker_legend_ax,
                    tree_genes_color=self.tree_genes_color,
                    marker_genes_color=self.marker_genes_color,
                    non_adata_celltypes_color=self.non_adata_celltypes_color,
                    i=i,
                )
                return_ax_dict[f"marker_legend_ax_{i}"] = marker_legend_ax

        if self.show_size_legend:
            size_legend_ax = fig.add_subplot(legend_gs[7])
            self._plot_size_legend(size_legend_ax)
            return_ax_dict["size_legend_ax"] = size_legend_ax

        if self.show_colorbar:
            color_legend_ax = fig.add_subplot(legend_gs[9])
            self._plot_colorbar(color_legend_ax, normalize)
            return_ax_dict["color_legend_ax"] = color_legend_ax
