"""
Battery metrics - compute the metrics of the battery materials
"""
from pymatgen.core import Composition
import numpy as np
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
    has_adjust_text = True
except ImportError:
    has_adjust_text = False


class TheoreticalCathode:
    """
    Class for theoretical cathode, used for extracting current/energy capacities
    """
    FARADAY_CONSTANT = 96485.3329  # Faraday constant in s A /mol

    def __init__(
            self,
            comp=None,
            structure=None,
            nli=None,
            avg_voltage=None,
            cycle_factor=1.0,
            label=None,
            plot_args=None,
    ):
        """
        Instantiate a theoretical cathode.
        Args:
            comp (Composition, str): Composition of the lithiated cathode
            nli (float): Number of Li that can be extracted per composition
        """
        self._label = label
        self.structure = structure
        if isinstance(comp, str):
            comp = Composition(comp)
        self.comp = comp if comp else structure.composition.reduced_composition
        self.nli = nli
        self.molar_weight = self.comp.weight
        self.avg_voltage = avg_voltage
        self.cycle_factor = cycle_factor
        self.plot_args = {'c': 'k', 'marker': '^'}
        if plot_args:
            self.plot_args.update(plot_args)

    @property
    def mAhg(self):
        """Gravimetric current capacity in mAh g-1"""
        return self.nli * self.FARADAY_CONSTANT / self.molar_weight / 3600 * 1000 * self.cycle_factor

    @property
    def mAhcm3(self):
        """Volumetric current capaciy in mAhcm-3"""
        return self.mAhg * self.structure.density

    @property
    def Whkg(self):
        """Gravimetric energy density in Whkg"""
        return self.mAhg * self.avg_voltage

    @property
    def WhL(self):
        return self.mAhcm3 * self.avg_voltage

    @property
    def formula(self):
        return self.comp.reduced_formula

    @property
    def label(self):
        return self._label if self._label else self.formula

    def __repr__(self):
        string = 'Cathode:{}, with maximum {} Li extaction ({} possible in practice)'.format(self.comp.reduced_formula, self.nli,
                                                                                             self.cycle_factor)
        return string


class Plotter:
    """
    Plotter for plotting figures showing the energy density of various cathodes
    """

    def __init__(self, entries, show_cycle_li=True):
        """
        Instantiate the object using a list of `TheoreticalCathode` entries.
        """
        self.entries = entries
        self.show_cycle_li = show_cycle_li

    def plot_whkg_mahg(self, adjust_text_kwargs=None):
        """
        Generate the gravimetric plot.
        """
        data = [[entry.mAhg, entry.Whkg] for entry in self.entries]
        texts = []
        for entry, (x, y) in zip(self.entries, data):
            label = entry.label
            if self.show_cycle_li and entry.cycle_factor != 1.0:
                label += ' ({:.2f} Li)'.format(entry.nli * entry.cycle_factor)
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y, **entry.plot_args)
        self._call_adjust_text(texts, adjust_text_kwargs)
        plt.xlabel(r'Current Capacity $\mathrm{(mAh g^{-1})}$')
        plt.ylabel(r'Energy Capacity $\mathrm{(Wh kg^{-1})}$')

    def _call_adjust_text(self, texts, adjust_text_kwargs=None):
        """Call the adjust_text function to adjust the positions of the labels"""
        if has_adjust_text:
            base_kwargs = dict(force_text=(0.2, 0.5),
                               only_move={
                                   'points': ('y', 'x'),
                                   'texts': ('y', 'x')
                               },
                               arrowprops=dict(arrowstyle='->', color='r', lw=1.0))
            if adjust_text_kwargs:
                base_kwargs.update(adjust_text_kwargs)
            adjust_text(texts, **base_kwargs)

    def plot_whl_mahcm3(self, adjust_text_kwargs=None):
        """
        Generate the volumemetric plot.
        """
        data = [[entry.mAhcm3, entry.WhL] for entry in self.entries]
        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1], 'o')
        texts = []
        for entry, (x, y) in zip(self.entries, data):
            label = entry.label
            if self.show_cycle_li and entry.cycle_factor != 1.0:
                label += ' ({:.2f} Li)'.format(entry.nli * entry.cycle_factor)
            label = entry.label + '({:.2f} Li)'.format(entry.nli * entry.cycle_factor)
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y, **entry.plot_args)
        self._call_adjust_text(texts, adjust_text_kwargs)
        plt.xlabel(r'Current Capacity $\mathrm{(mAh cm^{-3})}$')
        plt.ylabel(r'Energy Capacity $\mathrm{(Wh L^{-1})}$')

    @staticmethod
    def plot_gruide_lines(end=1200):
        plt.plot([0, end], [0, end * 2], label='2 V')
        plt.plot([0, end], [0, end * 3], label='3 V')
        plt.plot([0, end], [0, end * 4], label='4 V')
        plt.plot([0, end], [0, end * 5], label='5 V')
