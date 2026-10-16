import numpy as np
import pytest

from helpers import frame


@pytest.fixture
def random_context():
    rng = np.random.default_rng(12)
    return frame(np.exp(rng.normal(size=(10, 2)) * 0.28))
