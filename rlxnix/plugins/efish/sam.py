import nixio
import numpy as np
import matplotlib.pyplot as plt

from .efish_ephys_repro import EfishEphys


class Sam(EfishEphys):
    _repro_name = "SAM"

    def __init__(self, repro_run: nixio.Tag, traces, relacs_nix_version=1.1):
        super().__init__(repro_run, traces, relacs_nix_version=relacs_nix_version)

    @property
    def deltafs(self):
        """ The difference frequencies of the stimuli in Hertz.

        Returns
        -------
        dfs: list of floats
            The difference frequencies of the stimulus presentations in Hertz.
        """
        dfs = [s.metadata[s.name]["DeltaF"][0][0] for s in self.stimuli]
        return dfs

    @property
    def pause(self):
        """The pause between stimuli in seconds.

        Returns
        -------
        p: float
            The pause between stimuli in seconds.
        """
        d = self.metadata["RePro-Info"]["settings"]["pause"][0][0]
        return d

    @property
    def contrast(self):
        """The contrast of the stimuli, i.e. the stimulus amplitude relative to the EOD amplitude.

        Returns
        -------
        a: float
            The contrast.
        unit: str
        """
        a = self.metadata["RePro-Info"]["settings"]["amplitude"][0][0]
        unit = self.metadata["RePro-Info"]["settings"]["amplitude"][1]
        return a, unit

    @property
    def phase(self):
        """The phase of with which the stimulus starts in radians.

        Returns
        -------
        p: float
            The phase of the stimulus start in radians.
        """
        p = self.metadata["RePro-Info"]["settings"]["phase"][0][0]
        return p

    @property
    def is_sinewave(self):
        """True if the stimulus is a sinewave, False if it.

        Returns
        -------
        bool: whether or not the stimulus was a sinewave or something else.
        """
        return self.metadata["RePro-Info"]["settings"]["sinewave"][0][0]

    @property
    def is_amplitude_modulation(self):
        """True if the stimulus is amplitude modulated, False if it is not.

        Returns
        -------
        bool: whether the stimulus is an amplitude modulation or a direct stimulus.
        """
        return self.metadata["RePro-Info"]["settings"]["am"][0][0]