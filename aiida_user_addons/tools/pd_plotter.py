"""
Extending the pymatgen phase diagram module

Not full tested on latest versions of pymatgen
"""
from typing import Tuple
import json
import math
from pathlib import Path

import numpy as np
from pymatgen.core.composition import get_el_sp, gcd, formula_double_format, Composition

from pymatgen.analysis.phase_diagram import (CompoundPhaseDiagram, PDPlotter, order_phase_diagram, plotly_layouts, go, triangular_coord,
                                             order_phase_diagram, latexify)

with open(Path(__file__).parent / 'plotly_layouts.json') as f:
    plotly_layouts = json.load(f)


def pretty_plot(width=8, height=None, plt=None, dpi=None, color_cycle=('qualitative', 'Set1_9')):
    """
    Provides a publication quality plot, with nice defaults for font sizes etc.

    Args:
        width (float): Width of plot in inches. Defaults to 8in.
        height (float): Height of plot in inches. Defaults to width * golden
            ratio.
        plt (matplotlib.pyplot): If plt is supplied, changes will be made to an
            existing plot. Otherwise, a new plot will be created.
        dpi (int): Sets dot per inch for figure. Defaults to 300.
        color_cycle (tuple): Set the color cycle for new plots to one of the
            color sets in palettable. Defaults to a qualitative Set1_9.

    Returns:
        Matplotlib plot object with properly sized fonts.
    """
    ticksize = int(width * 2.5)

    golden_ratio = (math.sqrt(5) - 1) / 2

    if not height:
        height = int(width * golden_ratio)

    if plt is None:
        import matplotlib.pyplot as plt
        import importlib
        mod = importlib.import_module('palettable.colorbrewer.%s' % color_cycle[0])
        colors = getattr(mod, color_cycle[1]).mpl_colors
        from cycler import cycler

        plt.figure(figsize=(width, height), facecolor='w', dpi=dpi)
        ax = plt.gca()
        ax.set_prop_cycle(cycler('color', colors))
    else:
        fig = plt.gcf()
        fig.set_size_inches(width, height)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    ax = plt.gca()
    ax.set_title(ax.get_title(), size=width * 4)

    labelsize = int(width * 3)

    ax.set_xlabel(ax.get_xlabel(), size=labelsize)
    ax.set_ylabel(ax.get_ylabel(), size=labelsize)

    return plt


class BetterMplPlotter(PDPlotter):
    """
    A better version for plotting with matplotlib
    """

    def get_plot(
            self,
            label_stable=True,
            label_unstable=True,
            ordering=None,
            energy_colormap=None,
            process_attributes=False,
            plt=None,
            label_uncertainties=False,
            stable_label_size=24.,
            manual_stable_offsets=None,
            manual_unstable_offsets=None,
            unstable_label_size=18,
            no_polyanion=False,
    ):
        """
        :param label_stable: Whether to label stable compounds.
        :param label_unstable: Whether to label unstable compounds.
        :param ordering: Ordering of vertices (matplotlib backend only).
        :param energy_colormap: Colormap for coloring energy (matplotlib backend only).
        :param process_attributes: Whether to process the attributes (matplotlib
            backend only).
        :param plt: Existing plt object if plotting multiple phase diagrams (
            matplotlib backend only).
        :param label_uncertainties: Whether to add error bars to the hull (plotly
            backend only). For binaries, this also shades the hull with the
            uncertainty window.
        :return: go.Figure (plotly) or matplotlib.pyplot (matplotlib)
        """
        fig = None

        if self.backend == 'plotly':
            raise NotImplementedError
        elif self.backend == 'matplotlib':
            if self._dim <= 3:
                fig = self._get_2d_plot(
                    label_stable,
                    label_unstable,
                    ordering,
                    energy_colormap,
                    plt=plt,
                    process_attributes=process_attributes,
                    manual_stable_offsets=manual_stable_offsets,
                    stable_label_size=stable_label_size,
                    manual_unstable_offsets=manual_unstable_offsets,
                    unstable_label_size=unstable_label_size,
                    no_polyanion=no_polyanion,
                )
            elif self._dim == 4:
                fig = self._get_3d_plot(label_stable)

        return fig

    def _get_2d_plot(
            self,
            label_stable=True,
            label_unstable=True,
            ordering=None,
            energy_colormap=None,
            vmin_mev=-60.0,
            vmax_mev=60.0,
            show_colorbar=True,
            process_attributes=False,
            plt=None,
            manual_stable_offsets=None,
            stable_label_size=24,
            manual_unstable_offsets=None,
            unstable_label_size=18,
            no_polyanion=False,
    ):
        """
        Shows the plot using pylab. Contains import statements since matplotlib is a
        fairly extensive library to load.

        Args:
            manual_stable_offsets (None): Manual offsets for labeling stable entries
        """
        if plt is None:
            plt = pretty_plot(8, 6)
        from matplotlib.font_manager import FontProperties

        if ordering is None:
            (lines, labels, unstable) = self.pd_plot_data
        else:
            (_lines, _labels, _unstable) = self.pd_plot_data
            (lines, labels, unstable) = order_phase_diagram(_lines, _labels, _unstable, ordering)

        if energy_colormap is None:
            if process_attributes:
                for x, y in lines:
                    plt.plot(x, y, 'k-', linewidth=3, markeredgecolor='k')
                # One should think about a clever way to have "complex"
                # attributes with complex processing options but with a clear
                # logic. At this moment, I just use the attributes to know
                # whether an entry is a new compound or an existing (from the
                #  ICSD or from the MP) one.
                for x, y in labels.keys():
                    if (labels[(x, y)].attribute is None or labels[(x, y)].attribute == 'existing'):
                        plt.plot(x, y, 'ko', **self.plotkwargs)
                    else:
                        plt.plot(x, y, 'k*', **self.plotkwargs)
            else:
                for x, y in lines:
                    plt.plot(x, y, 'ko-', **self.plotkwargs)
        else:
            from matplotlib.colors import Normalize, LinearSegmentedColormap
            from matplotlib.cm import ScalarMappable

            for x, y in lines:
                plt.plot(x, y, 'k-', markeredgecolor='k')
            vmin = vmin_mev / 1000.0
            vmax = vmax_mev / 1000.0
            if energy_colormap == 'default':
                mid = -vmin / (vmax - vmin)
                cmap = LinearSegmentedColormap.from_list(
                    'my_colormap',
                    [
                        (0.0, '#005500'),
                        (mid, '#55FF55'),
                        (mid, '#FFAAAA'),
                        (1.0, '#FF0000'),
                    ],
                )
            else:
                cmap = energy_colormap
            norm = Normalize(vmin=vmin, vmax=vmax)
            _map = ScalarMappable(norm=norm, cmap=cmap)
            _energies = [self._pd.get_equilibrium_reaction_energy(entry) for coord, entry in labels.items()]
            energies = [en if en < 0.0 else -0.00000001 for en in _energies]
            vals_stable = _map.to_rgba(energies)
            ii = 0
            if process_attributes:
                for x, y in labels.keys():
                    if (labels[(x, y)].attribute is None or labels[(x, y)].attribute == 'existing'):
                        plt.plot(x, y, 'o', markerfacecolor=vals_stable[ii], markersize=12)
                    else:
                        plt.plot(x, y, '*', markerfacecolor=vals_stable[ii], markersize=18)
                    ii += 1
            else:
                for x, y in labels.keys():
                    plt.plot(x, y, 'o', markerfacecolor=vals_stable[ii], markersize=15)
                    ii += 1

        font = FontProperties()
        #font.set_weight("bold")
        font.set_size(stable_label_size)

        # Sets a nice layout depending on the type of PD. Also defines a
        # "center" for the PD, which then allows the annotations to be spread
        # out in a nice manner.
        if len(self._pd.elements) == 3:
            plt.axis('equal')
            plt.xlim((-0.1, 1.2))
            plt.ylim((-0.1, 1.0))
            plt.axis('off')
            center = (0.5, math.sqrt(3) / 6)
        else:
            all_coords = labels.keys()
            miny = min([c[1] for c in all_coords])
            ybuffer = max(abs(miny) * 0.1, 0.1)
            plt.xlim((-0.1, 1.1))
            plt.ylim((miny - ybuffer, ybuffer))
            center = (0.5, miny / 2)
            plt.xlabel('Fraction', fontsize=28)
            if isinstance(self._pd, CompoundPhaseDiagram) and self._pd.normalize_terminals is False:
                plt.ylabel('Formation energy (eV/f.u)', fontsize=28)
            else:
                plt.ylabel('Formation energy (eV/atom)', fontsize=28)

        for ientry, coords in enumerate(sorted(labels.keys(), key=lambda x: -x[1])):
            entry = labels[coords]
            label = entry.name

            if no_polyanion is True:
                label = reduce_formula_no_polyanion(Composition(label).as_dict())[0]

            # The follow defines an offset for the annotation text emanating
            # from the center of the PD. Results in fairly nice layouts for the
            # most part.
            vec = np.array(coords) - center
            vec = vec / np.linalg.norm(vec) * 10 if np.linalg.norm(vec) != 0 else vec
            valign = 'bottom' if vec[1] > 0 else 'top'
            if vec[0] < -0.01:
                halign = 'right'
            elif vec[0] > 0.01:
                halign = 'left'
            else:
                halign = 'center'

            # Adds additional offsets
            if manual_stable_offsets:
                vec += manual_stable_offsets.get(label, [0, 0])

            if label_stable:
                if process_attributes and entry.attribute == 'new':
                    plt.annotate(
                        latexify(label),
                        coords,
                        xytext=vec,
                        textcoords='offset points',
                        horizontalalignment=halign,
                        verticalalignment=valign,
                        fontproperties=font,
                        color='g',
                    )
                else:
                    plt.annotate(
                        latexify(label),
                        coords,
                        xytext=vec,
                        textcoords='offset points',
                        horizontalalignment=halign,
                        verticalalignment=valign,
                        fontproperties=font,
                    )

        if self.show_unstable:
            font = FontProperties()
            font.set_size(unstable_label_size)
            energies_unstable = [self._pd.get_e_above_hull(entry) for entry, coord in unstable.items()]
            if energy_colormap is not None:
                energies.extend(energies_unstable)
                vals_unstable = _map.to_rgba(energies_unstable)
            ii = 0
            for ientry, (entry, coords) in enumerate(unstable.items()):
                ehull = self._pd.get_e_above_hull(entry)
                if ehull < self.show_unstable:
                    vec = np.array(coords) - center
                    vec = (vec / np.linalg.norm(vec) * 10 if np.linalg.norm(vec) != 0 else vec)
                    if manual_unstable_offsets:
                        vec += manual_unstable_offsets[ientry]
                    if no_polyanion:
                        label = reduce_formula_no_polyanion(Composition(entry.name).as_dict())[0]
                    else:
                        label = entry.name

                    if energy_colormap is None:
                        plt.plot(
                            coords[0],
                            coords[1],
                            'o',
                            linewidth=3,
                            markeredgecolor='xkcd:coral',
                            markerfacecolor='xkcd:coral',
                            markersize=4,
                        )
                    else:
                        plt.plot(
                            coords[0],
                            coords[1],
                            'x',
                            linewidth=3,
                            markeredgecolor='k',
                            markerfacecolor=vals_unstable[ii],
                            markersize=8,
                        )
                    if label_unstable:
                        plt.annotate(
                            latexify(label),
                            coords,
                            xytext=vec,
                            textcoords='offset points',
                            horizontalalignment=halign,
                            color='b',
                            verticalalignment=valign,
                            fontproperties=font,
                        )
                    ii += 1
        if energy_colormap is not None and show_colorbar:
            _map.set_array(energies)
            cbar = plt.colorbar(_map)
            cbar.set_label(
                'Energy [meV/at] above hull (in red)\nInverse energy ['
                'meV/at] above hull (in green)',
                rotation=-90,
                ha='left',
                va='center',
            )
        #f = plt.gcf()
        #f.set_size_inches((8, 6))
        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.07)
        return plt


class PrettyPlotlyPlotter(PDPlotter):
    """
    A prettier version of pymatgen plotly plotter

    In addition, allow differentiation of entries from multiple sources
    """
    ENTRY_TYPE_STRING = 'entry_type'
    marker_size_map = {'AIRSS': 8, 'MP': 8, None: 8}
    marker_symbol_map = {'AIRSS': 'diamond', 'MP': 'diamond-open', None: 'diamond'}
    marker_line_map = {'AIRSS': False, 'MP': False, None: False}
    stalbe_marker_shape_map = {'AIRSS': 'circle', 'MP': 'circle-open', None: 'circle'}
    label_color = 'red'
    label_size = 14
    base_label_offset = 0.0
    offset_2d = 0.005  # extra distance to offset label position for clarity
    offset_3d = 0.1  # extra distance to offset label position for clarity

    def get_plot(
            self,
            label_stable=True,
            label_unstable=True,
            ordering=None,
            energy_colormap=None,
            process_parameters=False,
            plt=None,
            label_uncertainties=False,
    ):
        """
            :param label_stable: Whether to label stable compounds.
            :param label_unstable: Whether to label unstable compounds.
            :param ordering: Ordering of vertices (matplotlib backend only).
            :param energy_colormap: Colormap for coloring energy (matplotlib backend only).
            :param process_parameters: Whether to process the attributes (matplotlib
                backend only).
            :param plt: Existing plt object if plotting multiple phase diagrams (
                matplotlib backend only).
            :param label_uncertainties: Whether to add error bars to the hull (plotly
                backend only). For binaries, this also shades the hull with the
                uncertainty window.
            :return: go.Figure (plotly) or matplotlib.pyplot (matplotlib)
        """
        fig = None

        if self.backend == 'plotly':
            data = [self._create_plotly_lines()]

            if self._dim == 3:
                data.append(self._create_plotly_ternary_support_lines())
                data.append(self._create_plotly_ternary_hull())

            # Markers and labels for each type
            stable_labels_plots = []
            stable_marker_plots = []
            unstable_marker_plots = []
            for i, entry_type in enumerate(self.unique_types):
                stable_labels_plot = self._create_plotly_stable_labels(label_stable, entry_type=entry_type)
                stable_marker_plot, unstable_marker_plot = self._create_plotly_markers(label_uncertainties,
                                                                                       entry_type=entry_type,
                                                                                       colorbar=True if i == 0 else False)
                stable_labels_plots.append(stable_labels_plot)
                stable_marker_plots.append(stable_marker_plot)
                unstable_marker_plots.append(unstable_marker_plot)

            if self._dim == 2 and label_uncertainties:
                data.append(self._create_plotly_uncertainty_shading(stable_marker_plot))

            data.extend(stable_labels_plots)
            data.extend(unstable_marker_plots)
            data.extend(stable_marker_plots)

            fig = go.Figure(data=data)
            fig.layout = self._create_plotly_figure_layout()

        elif self.backend == 'matplotlib':
            if self._dim <= 3:
                fig = self._get_2d_plot(
                    label_stable,
                    label_unstable,
                    ordering,
                    energy_colormap,
                    plt=plt,
                    process_parameters=process_parameters,
                )
            elif self._dim == 4:
                fig = self._get_3d_plot(label_stable)

        return fig

    @property
    def entry_types(self):
        """Type label of the entry"""
        return [entry.parameters.get(self.ENTRY_TYPE_STRING) for entry in self._pd.all_entries]

    @property
    def unique_types(self):
        """Unique labels"""
        return list(set(self.entry_types))

    def _create_plotly_stable_labels(self, label_stable=True, entry_type=None):
        """
        Creates a (hidable) scatter trace containing labels of stable phases.
        Contains some functionality for creating sensible label positions.

        Select only certain entries

        :return: go.Scatter (or go.Scatter3d) plot
        """
        x, y, z, text, textpositions = [], [], [], [], []
        stable_labels_plot = None
        min_energy_x = None

        energy_offset = -0.1 * self._min_energy + self.base_label_offset

        if self._dim == 2:
            min_energy_x = min(list(self.pd_plot_data[1].keys()), key=lambda c: c[1])[0]

        for coords, entry in self.pd_plot_data[1].items():
            if entry.composition.is_element:  # taken care of by other function
                continue

            # Skip entries that is not the type wanted
            if entry.parameters.get(self.ENTRY_TYPE_STRING) != entry_type:
                continue

            x_coord = coords[0]
            y_coord = coords[1]
            textposition = None

            if self._dim == 2:
                textposition = 'bottom left'
                if x_coord >= min_energy_x:
                    textposition = 'bottom right'
                    x_coord += self.offset_2d
                else:
                    x_coord -= self.offset_2d
                y_coord -= self.offset_2d
            elif self._dim == 3:
                textposition = 'middle center'
                if coords[0] > 0.5:
                    x_coord += self.offset_3d
                else:
                    x_coord -= self.offset_3d
                if coords[1] > 0.866 / 2:
                    y_coord -= self.offset_3d
                else:
                    y_coord += self.offset_3d

                z.append(self._pd.get_form_energy_per_atom(entry) + energy_offset)

            elif self._dim == 4:
                x_coord = x_coord - self.offset_3d
                y_coord = y_coord - self.offset_3d
                textposition = 'bottom right'
                z.append(coords[2])

            x.append(x_coord)
            y.append(y_coord)
            textpositions.append(textposition)

            comp = entry.composition
            if hasattr(entry, 'original_entry'):
                comp = entry.original_entry.composition

            formula = list(comp.reduced_formula)
            text.append(self._htmlize_formula(formula))

        visible = True
        if not label_stable or self._dim == 4:
            visible = 'legendonly'

        name = 'Labels (stable)' if entry_type is None else f'Labels (stable) - {entry_type}'
        plot_args = dict(
            text=text,
            textposition=textpositions,
            mode='text',
            name=name,
            hoverinfo='skip',
            opacity=1.0,
            visible=visible,
            showlegend=True,
            textfont_size=self.label_size,
            textfont_color=self.label_color,
        )

        if self._dim == 2:
            stable_labels_plot = go.Scatter(x=x, y=y, **plot_args)
        elif self._dim == 3:
            stable_labels_plot = go.Scatter3d(x=y, y=x, z=z, **plot_args)
        elif self._dim == 4:
            stable_labels_plot = go.Scatter3d(x=x, y=y, z=z, **plot_args)

        return stable_labels_plot

    @property
    def e_above_hull_range(self):
        """
        Return the range of the e_above_hull for all entires
        """
        engs = []
        for entry in self.pd_plot_data[2].keys():
            e_above_hull = round(self._pd.get_e_above_hull(entry), 3)
            engs.append(e_above_hull)
        return min(engs), min(max(engs), self.show_unstable)

    def _create_plotly_markers(self, label_uncertainties=False, entry_type=None, colorbar=True):
        """
        Creates stable and unstable marker plots for overlaying on the phase diagram.

        :return: Tuple of Plotly go.Scatter (or go.Scatter3d) objects in order: (
            stable markers, unstable markers)
        """

        def get_marker_props(coords, entries, stable=True):
            """Method for getting marker locations, hovertext, and error bars
            from pd_plot_data"""
            x, y, z, texts, energies, uncertainties = [], [], [], [], [], []

            for coord, entry in zip(coords, entries):
                energy = round(self._pd.get_form_energy_per_atom(entry), 3)

                entry_id = getattr(entry, 'entry_id', 'no ID')
                comp = entry.composition

                if hasattr(entry, 'original_entry'):
                    comp = entry.original_entry.composition

                formula = comp.reduced_formula
                clean_formula = self._htmlize_formula(formula)
                label = f'{clean_formula} ({entry_id}) <br> ' f'{energy} eV/atom'

                if not stable:
                    e_above_hull = round(self._pd.get_e_above_hull(entry), 3)
                    if e_above_hull > self.show_unstable:
                        continue
                    label += f' (+{e_above_hull} eV/atom)'
                    energies.append(e_above_hull)
                else:
                    uncertainty = 0
                    if hasattr(entry, 'correction_uncertainty_per_atom') and label_uncertainties:
                        uncertainty = round(entry.correction_uncertainty_per_atom, 4)
                        label += f'<br> (Error: +/- {uncertainty} eV/atom)'
                    uncertainties.append(uncertainty)
                    energies.append(energy)

                # Add the entry type to the hover data
                label += f'<br> entry_type: {entry.parameters.get(self.ENTRY_TYPE_STRING)}'

                texts.append(label)

                x.append(coord[0])
                y.append(coord[1])

                if self._dim == 3:
                    z.append(energy)
                elif self._dim == 4:
                    z.append(coord[2])

            return {
                'x': x,
                'y': y,
                'z': z,
                'texts': texts,
                'energies': energies,
                'uncertainties': uncertainties,
            }

        # Filter only the types wanted
        stable_coords, stable_entries = (
            [key for key, value in self.pd_plot_data[1].items() if value.parameters.get(self.ENTRY_TYPE_STRING) == entry_type],
            [value for key, value in self.pd_plot_data[1].items() if value.parameters.get(self.ENTRY_TYPE_STRING) == entry_type],
        )
        unstable_entries, unstable_coords = (
            [key for key, value in self.pd_plot_data[2].items() if key.parameters.get(self.ENTRY_TYPE_STRING) == entry_type],
            [value for key, value in self.pd_plot_data[2].items() if key.parameters.get(self.ENTRY_TYPE_STRING) == entry_type],
        )

        stable_props = get_marker_props(stable_coords, stable_entries)

        unstable_props = get_marker_props(unstable_coords, unstable_entries, stable=False)

        stable_markers, unstable_markers = dict(), dict()

        stable_name = 'Stable' if entry_type is None else f'Stable - {entry_type}'
        stable_marker_props = self.get_stable_marker(entry_type, self._dim, stable_props)

        unstable_name = 'Above hull' if entry_type is None else f'Above hull - {entry_type}'
        unstable_marker_props = self.get_unstable_marker(entry_type, self._dim, unstable_props, colorbar)

        if self._dim == 2:
            stable_markers = plotly_layouts['default_binary_marker_settings'].copy()
            stable_markers.update(
                dict(
                    x=list(stable_props['x']),
                    y=list(stable_props['y']),
                    name=stable_name,
                    marker=stable_marker_props,
                    opacity=0.9,
                    hovertext=stable_props['texts'],
                    error_y=dict(
                        array=list(stable_props['uncertainties']),
                        type='data',
                        color='gray',
                        thickness=2.5,
                        width=5,
                    ),
                ))

            unstable_markers = plotly_layouts['default_binary_marker_settings'].copy()
            unstable_markers.update(
                dict(
                    x=list(unstable_props['x']),
                    y=list(unstable_props['y']),
                    name=unstable_name,
                    marker=unstable_marker_props,
                    hovertext=unstable_props['texts'],
                ))

        elif self._dim == 3:
            stable_markers = plotly_layouts['default_ternary_marker_settings'].copy()
            stable_markers.update(
                dict(
                    x=list(stable_props['y']),
                    y=list(stable_props['x']),
                    z=list(stable_props['z']),
                    name=stable_name,
                    marker=stable_marker_props,
                    hovertext=stable_props['texts'],
                    error_z=dict(
                        array=list(stable_props['uncertainties']),
                        type='data',
                        color='darkgray',
                        width=10,
                        thickness=5,
                    ),
                ))

            unstable_markers = plotly_layouts['default_ternary_marker_settings'].copy()
            unstable_markers.update(
                dict(
                    x=unstable_props['y'],
                    y=unstable_props['x'],
                    z=unstable_props['z'],
                    name=unstable_name,
                    marker=unstable_marker_props,
                    hovertext=unstable_props['texts'],
                ))

        elif self._dim == 4:
            stable_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
            stable_markers.update(
                dict(
                    x=stable_props['x'],
                    y=stable_props['y'],
                    z=stable_props['z'],
                    name=stable_name,
                    marker=stable_marker_props,
                    hovertext=stable_props['texts'],
                ))

            unstable_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
            unstable_markers.update(
                dict(
                    x=unstable_props['x'],
                    y=unstable_props['y'],
                    z=unstable_props['z'],
                    name=unstable_name,
                    marker=unstable_marker_props,
                    hovertext=unstable_props['texts'],
                    visible='legendonly',
                ))

        stable_marker_plot = go.Scatter(**stable_markers) if self._dim == 2 else go.Scatter3d(**stable_markers)
        unstable_marker_plot = go.Scatter(**unstable_markers) if self._dim == 2 else go.Scatter3d(**unstable_markers)

        return stable_marker_plot, unstable_marker_plot

    def get_stable_marker(self, entry_type, dim, stable_props=None):
        """Marker properties for stable entries"""
        size = self.marker_size_map.get(entry_type, 8)
        symbol = self.stalbe_marker_shape_map.get(entry_type, 'circle')

        if dim == 2:
            return dict(color='darkgreen', size=size, symbol=symbol, line=dict(color='black', width=2))
        if dim == 3:
            return dict(color='black', size=size, symbol=symbol, opacity=0.8, line=dict(color='black', width=2))
        if dim == 4:
            return dict(
                symbol=symbol,
                color=stable_props['energies'],
                colorscale=plotly_layouts['stable_markers_colorscale'],
                size=size,
                opacity=0.9,
            )

    def get_stable_color_scale(self):
        """Return the color scale to be used"""
        return plotly_layouts['stable_markers_colorscale']

    def get_unstable_color_scale(self):
        """Return the color scale to be used"""
        return plotly_layouts['unstable_markers_colorscale']

    def get_unstable_marker(self, entry_type, dim, unstable_props, colorbar=True):
        """Marker properties for unstable entries"""
        size = self.marker_size_map.get(entry_type, 8) - 2
        symbol = self.marker_symbol_map.get(entry_type, 'diamond')
        cmin, cmax = self.e_above_hull_range
        if dim == 2:
            out = dict(
                color=unstable_props['energies'],
                colorscale=plotly_layouts['unstable_colorscale'],
                size=size,
                symbol=symbol,
                cmin=cmin,
                cmax=cmax,
            )
        if dim == 3:
            out = dict(color=unstable_props['energies'],
                       colorscale=plotly_layouts['unstable_colorscale'],
                       size=size,
                       symbol=symbol,
                       cmin=cmin,
                       cmax=cmax,
                       colorbar=dict(title='Energy Above Hull<br>(eV/atom)', x=0.05, len=0.75))
        if dim == 4:
            out = dict(
                color=unstable_props['energies'],
                colorscale=plotly_layouts['unstable_colorscale'],
                size=size,
                symbol=symbol,
                colorbar=dict(title='Energy Above Hull<br>(eV/atom)', x=0.05, len=0.75),
                cmin=cmin,
                cmax=cmax,
            )
        # Add line if needed
        if self.marker_line_map.get(entry_type, False):
            out.update({'line': dict(color='blue', width=4)})

        # Remove colorbar is not needed
        if not colorbar:
            out.pop('colorbar', None)

        return out

    def _create_plotly_ternary_hull(self):
        """
        Creates shaded mesh plot for coloring the ternary hull by formation energy.

        :return: go.Mesh3d plot
        """
        facets = np.array(self._pd.facets)
        coords = np.array([triangular_coord(c) for c in zip(self._pd.qhull_data[:-1, 0], self._pd.qhull_data[:-1, 1])])
        energies = np.array([self._pd.get_form_energy_per_atom(e) for e in self._pd.qhull_entries])

        return go.Mesh3d(
            x=list(coords[:, 1]),
            y=list(coords[:, 0]),
            z=list(energies),
            i=list(facets[:, 1]),
            j=list(facets[:, 0]),
            k=list(facets[:, 2]),
            opacity=0.8,
            intensity=list(energies),
            colorscale=plotly_layouts['stable_colorscale'],
            colorbar=dict(title='Formation energy<br>(eV/atom)', x=0.9, len=0.75),
            hoverinfo='skip',
            lighting=dict(diffuse=0.0, ambient=1.0),
            name='Convex Hull (shading)',
            flatshading=True,
            showlegend=True,
        )

    @staticmethod
    def turn_off_above_hull(fig):
        return turn_off_above_hull(fig)


def turn_off_above_hull(fig):
    """
    Change the visibility of above hull items in a figure
    """
    for data in fig.data:
        if data.name and data.name.startswith('Above hull'):
            data['visible'] = 'legendonly'
    return fig


def reduce_formula_no_polyanion(sym_amt, iupac_ordering=False) -> Tuple[str, int]:
    """
    Helper method to reduce a sym_amt dict to a reduced formula and factor.
    Unlike the original pymatgen version, this function does not do any polyanion reduction

    Args:
        sym_amt (dict): {symbol: amount}.
        iupac_ordering (bool, optional): Whether to order the
            formula by the iupac "electronegativity" series, defined in
            Table VI of "Nomenclature of Inorganic Chemistry (IUPAC
            Recommendations 2005)". This ordering effectively follows
            the groups and rows of the periodic table, except the
            Lanthanides, Actanides and hydrogen. Note that polyanions
            will still be determined based on the true electronegativity of
            the elements.

    Returns:
        (reduced_formula, factor).
    """
    syms = sorted(sym_amt.keys(), key=lambda x: [get_el_sp(x).X, x])

    syms = list(filter(lambda x: abs(sym_amt[x]) > Composition.amount_tolerance, syms))

    factor = 1
    # Enforce integers for doing gcd.
    if all((int(i) == i for i in sym_amt.values())):
        factor = abs(gcd(*(int(i) for i in sym_amt.values())))

    polyanion = []

    syms = syms[:len(syms) - 2 if polyanion else len(syms)]

    if iupac_ordering:
        syms = sorted(syms, key=lambda x: [get_el_sp(x).iupac_ordering, x])

    reduced_form = []
    for s in syms:
        normamt = sym_amt[s] * 1.0 / factor
        reduced_form.append(s)
        reduced_form.append(formula_double_format(normamt))

    reduced_form = ''.join(reduced_form + polyanion)
    return reduced_form, factor
