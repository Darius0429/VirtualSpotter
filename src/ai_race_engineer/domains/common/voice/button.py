from abc import ABC, abstractmethod


class Button(ABC):
    """Common interface for all talk buttons."""

    @abstractmethod
    def pressed(self) -> bool:
        """Whether the talk button is down now; never blocks.
        In:  -
        Out: bool
        """
