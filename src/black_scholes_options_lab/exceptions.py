"""Project-specific exceptions."""


class OptionsLabError(Exception):
    """Base exception for recoverable options lab errors."""


class MarketDataError(OptionsLabError):
    """Raised when market data cannot be retrieved or parsed."""


class InvalidOptionInputError(OptionsLabError, ValueError):
    """Raised when an option input is outside the model domain."""
