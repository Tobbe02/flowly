import numpy as np
import pytest

from keras.layers import (
    merge,

    Activation,
    Dense,
    Input,
    InputLayer,
    Lambda,
    Masking,
)
from keras.models import Model, Sequential

from flowly.ml.layers import build_recurrence_wrapper, GlobalAveragePooling1D, RecurrentWrapper


def test_recurrent_wrapper__simple_rnn__sequences():
    def recurrent_layer():
        hidden = Input((128,))
        input = Input((10,))

        x = Dense(128, activation='relu')(input)
        x = merge([hidden, x], mode='sum')
        new_hidden = Activation('sigmoid')(x)

        return RecurrentWrapper(
            input=[input],
            output=[new_hidden],
            bind={hidden: new_hidden},
            return_sequences=True,
        )

    m = Sequential([
        InputLayer(input_shape=(None, 10)),
        recurrent_layer(),
    ])

    result = m.predict(np.random.uniform(size=(30, 20, 10)))

    # map into hidden state
    assert result.shape == (30, 20, 128)


def test_recurrent_wrapper__simple_rnn__no_sequences():
    """Return only the latest step in the sequence
    """
    def recurrent_layer():
        hidden = Input((128,))
        input = Input((10,))

        x = Dense(128, activation='relu')(input)
        x = merge([hidden, x], mode='sum')
        new_hidden = Activation('sigmoid')(x)

        return RecurrentWrapper(
            input=[input],
            output=[new_hidden],
            bind={hidden: new_hidden},
            return_sequences=False,
        )

    m = Sequential([
        InputLayer(input_shape=(None, 10)),
        recurrent_layer(),
    ])

    result = m.predict(np.random.uniform(size=(30, 20, 10)))

    # map into hidden state
    assert result.shape == (30, 128)


def test_recurrent_wrapper__simple_rnn__output_model():
    """The hidden state is not returned, but mapped by an output model.
    """
    def recurrent_layer():
        hidden = Input((128,))
        input = Input((10,))

        x = Dense(128, activation='relu')(input)
        x = merge([hidden, x], mode='sum')
        new_hidden = Activation('sigmoid')(x)
        output = Dense(64)(x)

        return RecurrentWrapper(
            input=[input],
            output=[output],
            bind={hidden: new_hidden},
            return_sequences=True,
        )

    m = Sequential([
        InputLayer(input_shape=(None, 10)),
        recurrent_layer(),
    ])

    assert m.predict(np.random.uniform(size=(30, 20, 10))).shape == (30, 20, 64)


def test_recurrent_wrapper__multiple_outputs():
    def build_recurrent():
        hidden = Input((128,))
        input = Input((10,))

        x = Dense(128, activation='relu')(input)
        x = merge([hidden, x], mode='sum')
        new_hidden = Activation('sigmoid')(x)
        output = Dense(64)(new_hidden)

        return RecurrentWrapper(
            input=[input],
            output=[new_hidden, output],
            bind={hidden: new_hidden},
            return_sequences=True,
        )

    x = Input((None, 10))
    u, v = build_recurrent()(x)
    m = Model(input=[x], output=[u, v])

    u, v = m.predict(np.random.uniform(size=(30, 20, 10)))
    assert u.shape == (30, 20, 128)
    assert v.shape == (30, 20, 64)


def test_recurrent_wrapper__multiple_inputs():
    def build_recurrent():
        hidden = Input((128,))
        input_x = Input((10,))
        input_y = Input((5,))

        x = Dense(128, activation='relu')(input_x)
        y = Dense(128, activation='relu')(input_y)
        x = merge([hidden, x, y], mode='sum')
        new_hidden = Activation('sigmoid')(x)

        return RecurrentWrapper(
            input=[input_x, input_y],
            output=[new_hidden],
            bind={hidden: new_hidden},
            return_sequences=True,
        )

    x = Input((None, 10))
    y = Input((None, 5))
    u = build_recurrent()([x, y])

    m = Model(input=[x, y], output=[u])
    actual = m.predict([
        np.random.uniform(size=(30, 20, 10)),
        np.random.uniform(size=(30, 20, 5))
    ])

    assert actual.shape == (30, 20, 128)


def test_recurrent_wrapper__sequence_inputs():
    """The hidden state is not returned, but mapped by an output model.
    """
    def build_recurrent():
        input_current = Input(shape=(10,))
        input_seq = Input(shape=(None, 10))

        hidden = Input(shape=(128,))

        seq = GlobalAveragePooling1D()(input_seq)
        new_hidden = merge([
            Dense(128, activation='relu')(seq),
            Dense(128, activation='relu')(input_current),
            hidden
        ], mode='sum')
        new_hidden = Activation('sigmoid')(new_hidden)

        return RecurrentWrapper(
            input=[input_current],
            sequence_input=[input_seq],
            output=[new_hidden],
            bind={
                hidden: new_hidden
            },
            return_sequences=True,
        )

    seq = Input((None, 10))
    output = build_recurrent()([seq, seq])

    m = Model(input=[seq], output=[output])
    assert m.predict(np.random.uniform(size=(30, 20, 10))).shape == (30, 20, 128)


cases = [
    dict(
        desc='simple rnn',
        input=['x'],
        output=['new_hidden'],
        bind={
            'hidden': 'new_hidden',
        },
        state_input=['hidden'],
        state_output=['new_hidden'],
        final_output_map=[0],
    ),
    dict(
        desc='rnn with secondary output',
        input=['x'],
        output=['new_hidden', 'secondary'],
        bind={
            'hidden': 'new_hidden',
        },
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

