---
layout: "contents"
title: Training Minigrid Environments
firstpage:
---

# Training Minigrid Environments

The environments in the Minigrid library can be trained easily using [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/). In this tutorial we show how a PPO agent can be trained on the `MiniGrid-Empty-16x16-v0` environment.

## Create Custom Feature Extractor

Although `StableBaselines3` is fully compatible with `Gymnasium`-based environments (which includes Minigrid), the default CNN architecture does not directly support the Minigrid observation space. Thus, to train an agent on Minigrid environments, we need to create a custom feature extractor. This can be done by creating a feature extractor class that inherits from `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`

```python
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
```

This class is created based on the custom feature extractor [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor:~:text=Custom%20Feature%20Extractor-,%EF%83%81,-If%20you%20want), the CNN architecture is copied from Lucas Willems' [rl-starter-files](https://github.com/lcswillems/rl-starter-files/blob/317da04a9a6fb26506bbd7f6c7c7e10fc0de86e0/model.py#L18).

## Train a PPO Agent

The using the custom feature extractor, we can train a PPO agent on the `MiniGrid-Empty-16x16-v0` environment. The following code snippet shows how this can be done.

```python
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    normalize_images=False,
)

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(2e5)
```

By default the observation of Minigrid environments are dictionaries. Since the `CnnPolicy` from StableBaseline3 by default takes in image observations, we need to wrap the environment using the `ImgObsWrapper` from the Minigrid library. This wrapper converts the dictionary observation to an image observation.

## Further Reading

One can also pass dictionary observations to StableBaseline3 policies, for a walkthrough the process of doing so see [here](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations). An implementation utilizing this functionality can be found [here](https://github.com/BolunDai0216/MinigridMiniworldTransfer/blob/main/minigrid_gotoobj_train.py).
