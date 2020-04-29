import gym
import tensorflow as tf
from flatland.core.grid import grid4
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from models.common.models import NatureCNN, ImpalaCNN


class GlobalObsModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self._options = model_config['custom_options']
        self._model = GlobalObsModule(action_space=action_space, architecture=self._options['architecture'],
                                      name="global_obs_model", **self._options['architecture_options'])

    def forward(self, input_dict, state, seq_lens):
        obs = preprocess_obs(input_dict['obs'])
        logits, baseline = self._model(obs)
        self.baseline = tf.reshape(baseline, [-1])
        return logits, state

    def variables(self):
        return self._model.variables

    def value_function(self):
        return self.baseline


def preprocess_obs(obs) -> tf.Tensor:
    transition_map, agents_state, targets = obs

    processed_agents_state_layers = []
    for i, feature_layer in enumerate(tf.unstack(agents_state, axis=-1)):
        if i in {0, 1}:  # agent direction (categorical)
            feature_layer = tf.one_hot(tf.cast(feature_layer, tf.int32), depth=len(grid4.Grid4TransitionsEnum) + 1,
                                       dtype=tf.float32)
        elif i in {2, 4}:  # counts
            feature_layer = tf.expand_dims(tf.math.log(feature_layer + 1), axis=-1)
        else:  # well behaved scalars
            feature_layer = tf.expand_dims(feature_layer, axis=-1)
        processed_agents_state_layers.append(feature_layer)

    return tf.concat(
        [tf.cast(transition_map, tf.float32), tf.cast(targets, tf.float32)] + processed_agents_state_layers, axis=-1)


class GlobalObsModule(tf.Module):
    def __init__(self, action_space, architecture: str, name=None, **kwargs):
        super().__init__(name=name)
        assert isinstance(action_space, gym.spaces.Discrete), \
            "Currently, only 'gym.spaces.Discrete' action spaces are supported."
        with self.name_scope:
            if architecture == 'nature':
                self._cnn = NatureCNN(activation_out=True, **kwargs)
            elif architecture == 'impala':
                self._cnn = ImpalaCNN(activation_out=True, **kwargs)
            else:
                raise ValueError(f"Invalid architecture: {architecture}.")
            self._logits_layer = tf.keras.layers.Dense(units=action_space.n)
            self._baseline_layer = tf.keras.layers.Dense(units=1)

    def __call__(self, spatial_obs, non_spatial_obs=None):
        latent_repr = self._cnn(spatial_obs)
        logits = self._logits_layer(latent_repr)
        baseline = self._baseline_layer(latent_repr)
        return logits, baseline