class EMA:
    """
    Accumulates values over time, calculating an exponential moving average over time
    """

    _last_ema_value: float | None
    _current_value: float | None
    _current_ema_value: float | None
    _smooth_factor: float
    _num_samples: int
    _num_samples_collected: int

    def __init__(self, num_samples: int, smooth: float = 2):
        # A window of 0 leaves the instance permanently unusable rather than merely
        # useless: add() saturates the sample count at num_samples, so with 0 the count
        # never leaves 0 and ema_value raises however many samples arrive. Refusing the
        # window here reports the mistake at construction instead of at a distant read.
        if num_samples < 1:
            raise ValueError(
                f"num_samples must be at least 1 to average over; got {num_samples}")

        self._smooth_factor = smooth
        self._last_ema_value = None
        self._current_value = None
        self._current_ema_value = None
        self._num_samples = num_samples
        self._num_samples_collected = 0

    @property
    def has_samples(self) -> bool:
        """Whether ema_value can be read yet.

        Lets a diagnostic report "no average yet" instead of raising, so a log line
        cannot abort the work it is describing.
        """
        return self._num_samples_collected > 0

    @property
    def ema_value(self) -> float:
        """Current exponential moving average"""
        if self._num_samples_collected == 0:
            raise ValueError("Cannot calculate EMA until at least one sample is collected")

        if self._current_ema_value is None:
            scalar = self._smooth_factor / (1 + self._num_samples_collected)
            self._current_ema_value = (self._current_value or 0) * scalar
            self._current_ema_value += (self._last_ema_value or 0) * (1.0 - scalar)

        return self._current_ema_value

    def add(self, value: float):
        self._last_ema_value = self.ema_value if self._num_samples_collected > 0 else value
        self._num_samples_collected = (
            self._num_samples_collected + 1
            if self._num_samples_collected < self._num_samples
            else self._num_samples
        )
        self._current_value = value
        self._current_ema_value = None

    def reset(self, value: float):
        """Reset the EMA to the specified value"""
        self._last_ema_value = value
        self._num_samples_collected = self._num_samples
        self._current_value = value
        self._current_ema_value = value
