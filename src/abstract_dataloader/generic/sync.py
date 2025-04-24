"""Generic Time Synchronization Protocols."""

import numpy as np
from jaxtyping import Float64, UInt32

from abstract_dataloader import spec


class Empty(spec.Synchronization):
    """Dummy synchronization which does not synchronize sensor pairs.

    No samples will be registered, and the trace can only be used as a
    collection of sensors.
    """

    def __call__(
        self, timestamps: dict[str, Float64[np.ndarray, "_N"]]
    ) -> dict[str, UInt32[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: input sensor timestamps.

        Returns:
            Synchronized index map.
        """
        return {k: np.array([], dtype=np.uint32) for k in timestamps}


class Next(spec.Synchronization):
    """Next sample synchronization, with respect to a reference sensor.

    Applies the following:

    - Find the start time, defined by the earliest time which is observed by
      all sensors, and the end time, defined by the last time which is observed
      by all sensors.
    - Truncate the reference sensor's timestamps to this start and end time,
      and use this as the query timestamps.
    - For each time in the query, find the first sample from each sensor which
      is after this time.

    See [`Synchronization`][abstract_dataloader.spec.] for protocol details.

    Args:
        reference: reference sensor to synchronize to.
    """

    def __init__(self, reference: str) -> None:
        self.reference = reference

    def __call__(
        self, timestamps: dict[str, Float64[np.ndarray, "_N"]]
    ) -> dict[str, UInt32[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: input sensor timestamps.

        Returns:
            Synchronized index map.
        """
        try:
            ref_time_all = timestamps[self.reference]
        except KeyError:
            raise KeyError(
                f"Reference sensor {self.reference} was not provided in "
                f"timestamps, with keys: {list(timestamps.keys())}")

        start_time = max(t[0] for t in timestamps.values())
        end_time = min(t[-1] for t in timestamps.values())

        start_idx = np.searchsorted(ref_time_all, start_time)
        end_idx = np.searchsorted(ref_time_all, end_time)
        ref_time = ref_time_all[start_idx:end_idx]
        return {
            k: np.searchsorted(v, ref_time).astype(np.uint32)
            for k, v in timestamps.items()}


class Nearest(spec.Synchronization):
    """Nearest sample synchronization, with respect to a reference sensor.

    Applies the following:

    - Compute the midpoints between observations between each sensor.
    - Find which bin the reference sensor timestamps fall into.
    - Calculate the resulting time delta between timestamps. If this exceeds
      `tol` for any sensor-reference pair, remove this match.

    See [`Synchronization`][abstract_dataloader.spec.] for protocol details.

    Args:
        reference: reference sensor to synchronize to.
        tol: synchronization time tolerance, in seconds. Setting `tol = np.inf`
            works to disable this check altogether.
    """

    def __init__(self, reference: str, tol: float = 0.1) -> None:
        if tol < 0:
            raise ValueError(
                f"Synchronization tolerance must be positive: {tol} < 0")

        self.tol = tol
        self.reference = reference

    def __call__(
        self, timestamps: dict[str, Float64[np.ndarray, "_N"]]
    ) -> dict[str, UInt32[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: input sensor timestamps.

        Returns:
            Synchronized index map.
        """
        try:
            t_ref = timestamps[self.reference]
        except KeyError:
            raise KeyError(
                f"Reference sensor {self.reference} was not provided in "
                f"timestamps, with keys: {list(timestamps.keys())}")

        indices = {
            k: np.searchsorted(
                (t_sensor[:-1] + t_sensor[1:]) / 2, t_ref
            ).astype(np.uint32)
            for k, t_sensor in timestamps.items()}
        valid = np.all(np.array([
           np.abs(timestamps[k][i_nearest] - t_ref) < self.tol
        for k, i_nearest in indices.items()]), axis=0)

        return {k: v[valid] for k, v in indices.items()}
