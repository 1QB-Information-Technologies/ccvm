# Python 2/3 compatibility
from future.utils import iteritems
import types


class StrDictMixIn(object):
    """String Dictionary MixIn Class.

    A mixin class that provides an __str__ method that returns a string of a
    dict containing all "public" attributes.
    """

    def __str__(self):
        """Overrides the default implementation."""
        d = {}
        for key, value in iteritems(self.__dict__):
            # Skip methods, internals and calleables
            if (
                isinstance(value, types.FunctionType)
                or key.startswith("_")
                or callable(value)
                or value is None
            ):
                continue

            d[key] = value

        return str(d)
