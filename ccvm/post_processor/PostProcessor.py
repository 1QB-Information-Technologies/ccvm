from abc import ABC, abstractmethod

class PostProcessor(ABC):
    """A Hypothetical PostProcessor Class Interface"""

    @abstractmethod
    def postprocess(self):
        """An abstract interface method"""
        pass
