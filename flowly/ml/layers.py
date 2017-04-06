from keras.engine import InputSpec
from keras.layers import GlobalAveragePooling1D as BaseGlobalAveragePooling1D, Layer, Input
from keras.engine.topology import Container

from keras import backend as K

import tensorflow as tf


class GlobalAveragePooling1D(BaseGlobalAveragePooling1D):
    """Global average pooling with masking support.
    """
    def __init__(self, **kwargs):
        super(GlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        if mask is None:
            return super(GlobalAveragePooling1D, self).call(x)

        mask = K.expand_dims(mask)
        mask = K.tile(mask, [1, 1, K.shape(x)[2]])
        mask = K.cast(mask, K.dtype(x))

        safe_mask_sum = K.sum(mask, axis=1)
        safe_mask_sum = K.maximum(safe_mask_sum, K.ones_like(safe_mask_sum))

        return K.sum(mask * x, axis=1) / safe_mask_sum


class ReplaceWithMaskLayer(Layer):
    def __init__(self, **kwargs):
        super(ReplaceWithMaskLayer, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask):
        return None

    def call(self, inputs, mask=None):
        if mask is None:
            mask = K.zeros_like(inputs)
            mask = K.sum(mask, axis=-1)
            mask = 1 + mask

        return K.expand_dims(mask)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3, "expected time-series data"
        return (input_shape[0], input_shape[1], 1)


class FixedLinspaceLayer(Layer):
    """A layer that ignores its input and returns a linspace with fixed length.
    """
    def __init__(self, start=0.0, stop=1.0, num=50, **kwargs):
        super(FixedLinspaceLayer, self).__init__(**kwargs)
        self.num = num
        self.start = float(start)
        self.stop = float(stop)

    def call(self, x):
        r = K.cast(K.arange(self.num), K.floatx()) / float(self.num - 1)
        r = self.start + (self.stop - self.start) * r
        r = K.expand_dims(K.expand_dims(r), axis=0)
        r = K.cast(r, dtype=K.floatx())
        r = K.tile(r, (K.shape(x)[0], 1, 1))
        return r

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3, "expected time-series data"
        return (input_shape[0], self.num, 1)


class LinspaceLayer(Layer):
    """Replace the input with a linearly spaced sequence.

    Supports masking.
    """
    def __init__(self, start=0.0, stop=1.0, **kwargs):
        super(LinspaceLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.start = start
        self.stop = stop

    def compute_mask(self, inputs, mask):
        return mask

    def call(self, inputs, mask=None):
        if mask is None:
            mask = K.zeros_like(inputs)
            mask = K.sum(mask, axis=-1)
            mask = 1 + mask

        else:
            mask = K.cast(mask, K.dtype(inputs))

        safe_n1 = K.sum(mask, axis=1) - 1
        safe_n1 = K.maximum(safe_n1, K.ones_like(safe_n1))
        safe_n1 = K.expand_dims(safe_n1)

        r = tf.cumsum(mask, axis=1) - 1
        r = self.start + (self.stop - self.start) * r / safe_n1
        r = mask * r
        r = K.expand_dims(r)
        return r

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3, "expected time-series data"
        return (input_shape[0], input_shape[1], 1)


class SoftmaxAttentionLayer(Layer):
    """Compute softmax attenion given a sequence and a context vector.

    Expect inputs with shapes:

    - sequence: ``(n_batch, n_time, n_features)``
    - context: ``(n_batch, n_features)``

    The output has then a shape ``(n_batch, n_features)``.
    """
    def __init__(self, epsilon=1e-7, **kwargs):
        super(SoftmaxAttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = epsilon

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shapes):
        seq_shape, _ = input_shapes
        return (seq_shape[0], seq_shape[2])

    def call(self, inputs, mask=None):
        seq, ctx = inputs

        if mask is None:
            mask = tf.ones(tf.shape(seq)[:2])

        else:
            mask, _ = mask

        # form the dot product between context vector and each item
        c = tf.expand_dims(ctx, axis=1)
        a = tf.reduce_sum(c * seq, axis=-1)

        # build the attention weights
        a = tf.cast(mask, K.floatx()) * tf.exp(a)
        a = a / tf.maximum(self.epsilon, tf.reduce_max(a, axis=1, keep_dims=True))
        a = a / tf.maximum(self.epsilon, tf.reduce_sum(a, axis=1, keep_dims=True))
        a = tf.expand_dims(a, axis=2)

        # compute the weighted sum
        return tf.reduce_sum(seq * a, axis=1)


class RecurrentWrapper(Layer):
    """Define a recurrent model in terms of a general transition model.

    A simple RNN for example would be expressed as:

        hidden = Input((128,))
        input = Input((10,))

        x = Dense(128, activation='relu')(input)
        x = Merge(mode='sum')([hidden, x])
        new_hidden = Activation('sigmoid')(x)

        rnn = RecurrentWrapper(
            input=[input],
            output=[new_hidden],
            bind={hidden: new_hidden},
            input_shape=(None, 10),
            return_sequences=True,
        )

    .. note::

        this class is still highly in flux and does only work with tensoflow atm.

    """
    def __init__(
            self, input, output, bind,
            sequence_input=(),
            stateful=False,
            return_sequences=False,
            **kwargs
    ):
        if stateful:
            raise RuntimeError('RecurrentWrapper does not support statefule transforms currently')

        self.supports_masking = True
        self.stateful = stateful
        self.return_sequences = return_sequences

        self.bindings = bind
        self.external_input = self._ensure_list(input)
        self.external_sequence_input = self._ensure_list(sequence_input)
        self.external_output = self._ensure_list(output)

        self.state_input, self.state_output, self.final_output_map = build_recurrence_wrapper(
            self.external_input + self.external_sequence_input,
            self.external_output,
            self.bindings,
            build_input,
        )

        self.step_model = Container(
            input=self.external_input + self.external_sequence_input + self.state_input,
            output=self.state_output,
        )

        self.losses = []
        super(RecurrentWrapper, self).__init__(**kwargs)

    @property
    def number_of_inputs(self):
        return len(self.external_input) + len(self.external_sequence_input)

    def compute_output_shape(self, input_shape):
        head = tuple(input_shape[:2] if self.return_sequences else input_shape[:1])

        def _shape(output):
            tail = (K.int_shape(output)[1],)
            return head + tail

        return [_shape(output) for output in self.external_output]

    def compute_mask(self, input, mask=None):
        return [
            (mask if self.return_sequences else None)
            for _ in self.external_output
        ]

    def get_constants(self, x):
        return []

    def reset_states(self):
        pass

    def get_config(self):
        return self.step_model.get_config()

    def build(self, input_shape):
        if self.number_of_inputs > 1:
            self.input_spec = [InputSpec(shape=shape) for shape in input_shape]

        else:
            self.input_spec = [InputSpec(shape=input_shape)]

        self.step_model.build(input_shape)

        self._trainable_weights.extend(self.step_model.trainable_weights)
        self._non_trainable_weights.extend(self.step_model.non_trainable_weights)
        self.constraints.update(self.step_model.constraints)
        self.losses.extend(self.step_model.losses)

        self.built = True

    def get_initial_states(self, x):
        return [
            self._get_initial_state(x, inp)
            for inp in self.state_input
        ]

    def call(self, x, mask=None):
        x = self._ensure_list(x)
        initial_states = self.get_initial_states(x)

        recurrent_inputs = x[:len(self.external_input)]
        sequence_inputs = x[len(self.external_input):]

        def step(states, inputs):
            full_input = self._ensure_list(inputs) + sequence_inputs + self._ensure_list(states)
            new_states = self.step_model.call(full_input)
            return self._ensure_list(new_states)

        recurrent_inputs = self._swap_time_and_samples(recurrent_inputs)

        outputs = tf.scan(step, recurrent_inputs, initial_states)
        outputs = [outputs[idx] for idx in self.final_output_map]

        if self.return_sequences:
            return self._possibly_scalar(self._swap_time_and_samples(outputs))

        return self._possibly_scalar([output[-1] for output in outputs])

    @staticmethod
    def _get_initial_state(x, inp):
        # TODO: check that all x have the same number of samples / timesteps
        # TODO: test that x has 3 dimensions and inp has two dimensions
        x = x[0]
        input_dim = int(inp.get_shape()[1])

        # copied from keras. Recurrent.get_initial_state
        initial_state = K.zeros_like(x, dtype=inp.dtype)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        return K.tile(initial_state, [1, input_dim])  # (samples, output_dim)

    @staticmethod
    def _swap_time_and_samples(inputs):
        return [
            tf.transpose(inp, [1, 0, 2])
            for inp in inputs
        ]

    @staticmethod
    def _ensure_list(obj):
        return list(obj) if isinstance(obj, (list, tuple)) else [obj]

    @staticmethod
    def _possibly_scalar(obj):
        return obj[0] if len(obj) == 1 else obj


def build_recurrence_wrapper(external_input, external_output, bind, build_input):
    """

    - the step model always returns the full state

    :return:
        state_input, state_output, final_output_map

    """
    state_input = []
    state_output = []

    bind = list(bind.items())

    for bound_input, bound_output in bind:
        if bound_input in external_input:
            raise ValueError('cannot bind an external input')

        state_input.append(bound_input)
        state_output.append(bound_output)

    for output in external_output:
        if output in state_output:
            continue

        state_input.append(build_input(output))
        state_output.append(output)

    final_output_map = [state_output.index(item) for item in external_output]

    return state_input, state_output, final_output_map


def build_input(output):
    return Input(output.shape[1:], dtype=output.dtype)
