import nixio
import numpy as np
from enum import Enum

from .mappings import DataType, type_map


class TimeReference(Enum):
    repro_start = 0
    zero = 1


class TraceContainer(object):
    def __init__(self, tag_or_mtag, index=None, relacs_nix_version=1.1) -> None:
        """[summary]

        """
        super().__init__()
        if isinstance(tag_or_mtag, nixio.MultiTag) and index is None:
            raise ValueError("Index must not be None, if a multiTag is passed!")

        self._tag = tag_or_mtag
        self._mapping_version = relacs_nix_version
        self._index = index 
        
        if isinstance(self._tag, nixio.MultiTag):
            self._start_time = self._tag.positions[self._index, 0][0]
            self._duration = self._tag.extents[self._index, 0][0] if self._tag.extents else 0.0
        else:
            self._start_time = self._tag.position[0]
            self._duration = self._tag.extent[0] if self._tag.extent else 0.0

    @property
    def name(self) -> str:
        """The name of the data segment

        Returns:
            string: the name
        """
        return self._tag.name

    @property
    def type(self) -> str:
        """The type of the data segment

        Returns:
            string: the type
        """
        return self._tag.type

    @property
    def start_time(self) -> float:
        """The start time of the 

        Returns:
            float: RePro start time
        """
        return self._start_time

    @property
    def duration(self) -> float:
        """The duration of the repro run in seconds.

        Returns:
            float: the duration in seconds.
        """
        return self._duration

    @property
    def repro_tag(self):
        """Returns the underlying tag

        Returns:
        --------
            nixio Tag or MultiTag: the tag
        """
        return self._tag

    @property
    def references(self) -> list:
        """The list of referenced event and data traces

        Returns:
            List: index, name and type of the references
        """
        refs = []
        for i, r in enumerate(self._tag.references):
            refs.append((i, r.name, r.type))
        return refs

    @property
    def features(self) -> list:
        """List of features associated with this repro run.

        Returns:
            List: name and type of t[description]
        """
        features = []
        for i, feats in enumerate(self._tag.features):
            features.append((i, feats.data.name, feats.data.type))
        return features

    def trace_data(self, name_or_index, reference=TimeReference.zero):
        """Get the data that was recorded while this repro was run.

        Args:
            name_or_index (str or int): name or index of the referenced data trace

        Returns:
            data (np.ndarray): the data 
            time (np.ndarray): the respective time vector, None, if the data is an event trace
        """
        ref = self._tag.references[name_or_index]
        time = None
        continuous_data_type = type_map[self._mapping_version][DataType.continuous] 
        data = ref.get_slice([self.start_time], [self.duration], nixio.DataSliceMode.Data)[:]
        start_position = self.start_time if reference is TimeReference.repro_start else 0.0

        if continuous_data_type in ref.type:  
            time = np.array(ref.dimensions[0].axis(len(data), start_position=start_position))
        else:  # event data
            data -= self.start_time
        return data, time

    def feature_data(self, name_or_index):
        feat_data = self._tag.feature_data(name_or_index)
        return feat_data[:]