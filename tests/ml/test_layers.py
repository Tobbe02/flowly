import numpy as np
import pytest
import tensorflow as tf

from keras.layers import (
    merge,

    Activation,
    Dense,
    Input,
    InputLayer,
    Masking,
)
from keras.models import Model, Sequential

from flowly.ml.layers import (
    build_recurrence_wrapper,
    FixedLinspaceLayer,
    GlobalAveragePooling1D,
    LinspaceLayer,
    RecurrentWrapper,
    ReplaceWithMaskLayer,
    SoftmaxAttentionLayer,
)


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


def test_fixed_linspace_layer__example():
    actual = apply_fixed_linspace(num=10, batch_input_shape=(5, 20, 3))
    expected = tiled_linspace(batch_input_shape=(5, 10, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_fixed_linspace_layer__other_range():
    actual = apply_fixed_linspace(num=10, start=2.0, stop=-1.0, batch_input_shape=(5, 20, 3))
    expected = tiled_linspace(start=2.0, stop=-1.0, batch_input_shape=(5, 10, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_replace_with_mask_layer__no_mask():
    m = Sequential([
        ReplaceWithMaskLayer(input_shape=(None, 10)),
    ])
    actual = m.predict(np.random.uniform(size=(30, 20, 10)))
    expected = np.ones((30, 20, 1))
    np.testing.assert_allclose(actual, expected)


def test_replace_with_mask_layer__with_mask():
    input = np.zeros((5, 20, 10))
    input[0, :10, :] = 1
    input[1, :20, :] = 1
    input[2, :15, :] = 1

    m = Sequential([
        Masking(input_shape=(None, 10)),
        ReplaceWithMaskLayer(),
    ])
    actual = m.predict(input)
    expected = input[:, :, :1]
    np.testing.assert_allclose(actual, expected)


def test_linspace_layer__no_mask():
    m = Sequential([
        LinspaceLayer(input_shape=(None, 10)),
    ])
    actual = m.predict(np.random.uniform(size=(30, 20, 10)))
    expected = tiled_linspace(batch_input_shape=(30, 20, 1))

    np.testing.assert_allclose(actual, expected)


def test_linspace_layer__with_mask():
    input = np.zeros((5, 20, 10))
    input[0, :10, :] = 1
    input[1, :20, :] = 1
    input[2, :15, :] = 1

    expected = np.zeros((5, 20, 1), dtype=float)
    expected[0, :10, 0] = np.linspace(0.0, 1.0, 10)
    expected[1, :20, 0] = np.linspace(0.0, 1.0, 20)
    expected[2, :15, 0] = np.linspace(0.0, 1.0, 15)

    m = Sequential([
        Masking(input_shape=(None, 10)),
        LinspaceLayer(),
    ])
    actual = m.predict(input)

    np.testing.assert_allclose(actual, expected)


def test_linspace_layer__with_mask_offset():
    input = np.zeros((5, 20, 10))
    input[0, 5:10, :] = 1
    input[1, 10:20, :] = 1
    input[2, 7:15, :] = 1

    expected = np.zeros((5, 20, 1), dtype=float)
    expected[0, 5:10, 0] = np.linspace(0.0, 1.0, 5)
    expected[1, 10:20, 0] = np.linspace(0.0, 1.0, 10)
    expected[2, 7:15, 0] = np.linspace(0.0, 1.0, 8)

    m = Sequential([
        Masking(input_shape=(None, 10)),
        LinspaceLayer(),
    ])
    actual = m.predict(input)

    np.testing.assert_allclose(actual, expected)


def tiled_linspace(batch_input_shape, start=0.0, stop=1.0):
    n_samples, num, n_features = batch_input_shape
    return np.tile(
        np.reshape(
            np.linspace(start, stop, num),
            (1, -1, 1),
        ),
        (n_samples, 1, n_features),
    )


def apply_fixed_linspace(**kwargs):
    batch_input_shape = kwargs.pop('batch_input_shape')

    m = Sequential([
        FixedLinspaceLayer(
            batch_input_shape=(None, None) + (batch_input_shape[-1],),
            **kwargs
        ),
    ])
    return m.predict(np.random.uniform(size=batch_input_shape))


def test_attention_no_masking():
    """Test without masking
    """
    s = np.random.uniform(size=(5, 4, 3)).astype(np.float32)
    c = np.random.uniform(size=(5, 3)).astype(np.float32)
    mask = np.ones((5, 4), dtype=np.bool)

    expected = _attention_expected(s, c, mask)
    actual = _attention_apply(s, c, mask)

    np.testing.assert_allclose(expected, actual, rtol=1e-3)


def test_attention_masking():
    """Test with masking (upto a fully masked array).
    """
    s = np.random.uniform(size=(5, 4, 3)).astype(np.float32)
    c = np.random.uniform(size=(5, 3)).astype(np.float32)

    mask = np.ones((5, 4), dtype=np.bool)
    mask[0, 0:] = 0
    mask[1, 1:] = 0
    mask[2, 2:] = 0
    mask[3, 3:] = 0
    mask[4, 4:] = 0

    expected = _attention_expected(s, c, mask)
    actual = _attention_apply(s, c, mask)

    np.testing.assert_allclose(expected, actual, rtol=1e-3)


def test_attention__invalid_masking():
    """test, that the mask is really handled correctly in refernce
    """
    s = np.random.uniform(size=(5, 4, 3)).astype(np.float32)
    c = np.random.uniform(size=(5, 3)).astype(np.float32)

    # note the slice is reversed
    mask = np.ones((5, 4), dtype=np.bool)
    mask[0, :0] = 0
    mask[1, :1] = 0
    mask[2, :2] = 0
    mask[3, :3] = 0
    mask[4, :4] = 0

    expected = _attention_expected(s, c, mask)
    actual = _attention_apply(s, c, mask)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(expected, actual, rtol=1e-3)


def _attention_apply(s, c, mask):
    with tf.Session() as sess:
        return (
            SoftmaxAttentionLayer()
            .call(
                [tf.constant(s), tf.constant(c)],
                mask=(tf.constant(mask), None),
            )
            .eval(session=sess)
        )


def _attention_expected(s, c, mask):
    N = s.shape[0]

    result = np.empty_like(c)

    for i in range(N):
        d = int(mask[i, :].sum())

        a = np.sum(s[i, :d, :] * c[i, None, :], axis=1)
        a = np.exp(a)
        a = a / np.sum(a, axis=0)[None]

        result[i, :] = np.sum(s[i, :d, :] * a[:, None], axis=0)

    return result
