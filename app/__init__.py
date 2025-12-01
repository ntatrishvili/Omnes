# App package initialization
# Suppress a noisy DeprecationWarning originating from SWIG-based extensions
# (e.g., certain pandapower/native extensions) that trigger on Python 3.12
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
    category=DeprecationWarning,
)

# Suppress repeated UserWarnings emitted by validation utilities during tests
warnings.filterwarnings(
    "ignore",
    message=r"Negative time index .* clamped to 0",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Time index .* exceeds array bounds",
    category=UserWarning,
)
