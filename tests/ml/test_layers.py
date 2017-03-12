import collections

import numpy as np
import pytest

from keras.layers import Masking
from keras.models import Sequential

from flowly.ml.layers import build_recurrence_wrapper, GlobalAveragePooling1D


def ordered(*items):
    return collections.OrderedDict(items)


cases = [
    dict(
        desc='simple rnn',
        input=['x'],
        output=['new_hidden'],
        bind=ordered(
            ('hidden', 'new_hidden'),
        ),
        state_input=['hidden'],
        state_output=['new_hidden'],
        final_output_map=[0],
    ),
    dict(
        desc='rnn with secondary output',
        input=['x'],
        output=['new_hidden', 'secondary'],
        bind=ordered(
            ('hidden', 'new_hidden'),
        ),
        state_input=['hidden', 'input-secondary'],
        state_output=['new_hidden', 'secondary'],
        final_output_map=[0, 1],
    )
]


@pytest.mark.parametrize('case', cases, ids=lambda c: c.get('desc'))
def test_build_recurrence_wrapper(case):
    actual = build_recurrence_wrapper(case['input'], case['output'], case['bind'], build_input)
    state_input, state_output, final_output_map = actual

    assert state_input == case['state_input']
    assert state_output == case['state_output']
    assert final_output_map == case['final_output_map']


def build_input(output):
    return 'input-' + output


def test_global_average_pooling_1d__no_masking():
    model = Sequential([GlobalAveragePooling1D(input_shape=(None, 20))])
    x_input = np.random.uniform(size=(40, 30, 20))

    result = model.predict(x_input)

    # use reduced tolerance due to float32
    np.testing.assert_allclose(result, np.mean(x_input, axis=1), rtol=1e-3)


def test_global_average_pooling_1d__with_masking():
    model = Sequential([
            Masking(input_shape=(None, 20)),
            GlobalAveragePooling1D()
    ])

    x_input = np.random.uniform(size=(40, 30, 20))
    x_input[:, -15:, :] = 0.0

    result = model.predict(x_input)

    # use reduced tolerance due to float32
    np.testing.assert_allclose(result, np.mean(x_input[:, :-15, :], axis=1), rtol=1e-3)
