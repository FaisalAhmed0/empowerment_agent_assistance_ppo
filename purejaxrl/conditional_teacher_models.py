import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import distrax
from flax.linen.initializers import orthogonal, constant

class ConditionalTeacherActorCritic(nn.Module):
    num_actions: int
    obs_dim: int
    competence_dim: int
    student_action_dim: int
    activation: str = "tanh"
    hidden_dim: int = 128
    use_encoders: bool = False
    add: bool = True
    concatenate: bool = False

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        offset = 0
        obs = x[..., offset : offset + self.obs_dim]
        offset += self.obs_dim
        competence = x[..., offset : offset + self.competence_dim]
        offset += self.competence_dim
        student_action = x[..., offset : offset + self.student_action_dim]
        offset += self.student_action_dim
        # import pdb; pdb.set_trace()
        ### Competence MLP
        competence_encoding = act(nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(competence)
        )
        competence_encoding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(competence_encoding)

        x = jnp.concatenate([obs, student_action], axis=-1)

        if self.add:
            actor_mean = act(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x) 
            ) + competence_encoding
            actor_mean = act(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(actor_mean) 
            ) + competence_encoding
            logits = nn.Dense(
                self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            pi = distrax.Categorical(logits=logits)
            critic = act(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x) 
            ) + competence_encoding
            critic = act(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(critic) 
            ) + competence_encoding
            value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        elif self.concatenate:
            actor_mean = act(
                jnp.concatenate(
                    [
                        nn.Dense(
                            self.hidden_dim,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(x),
                        competence_encoding,
                    ],
                    axis=-1,
                )
            )
            actor_mean = act(
                jnp.concatenate(
                    [
                        nn.Dense(
                            self.hidden_dim,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(actor_mean),
                        competence_encoding,
                    ],
                    axis=-1,
                )
            )
            logits = nn.Dense(
                self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            pi = distrax.Categorical(logits=logits)

            critic = act(
                jnp.concatenate(
                    [
                        nn.Dense(
                            self.hidden_dim,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(x),
                        competence_encoding,
                    ],
                    axis=-1,
                )
            )
            critic = act(
                jnp.concatenate(
                    [
                        nn.Dense(
                            self.hidden_dim,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(critic),
                        competence_encoding,
                    ],
                    axis=-1,
                )
            )
            value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        else:
            raise ValueError(
                "ConditionalTeacherActorCritic requires add=True or concatenate=True"
            )
        return pi, jnp.squeeze(value, axis=-1)
