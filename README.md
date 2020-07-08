
# Alpacka

This repository provides the implementation of agents introduced in the "Structure and randomness in planning and reinforcement learning" paper. We are happy to discuss any concerns in the issues.

## Quickstart

### Install

`pip install -e .[dev]`

The `-e` option allows you to make changes to the package that will be immediately visible in the virtualenv.

### Run locally

The entrypoint for running experiments with Alpacka resides in `alpacka/runner.py`. Command to run on the local machine, without mrunner:

```
python3 -m alpacka.runner \
    --config_file=configs/shooting_random_cartpole.gin \
    --config=Runner.episode_time_limit=50 \
    --config=Runner.n_envs=8 \
    --output_dir=./out
```

where

- `config_file` - Path to the Gin config file describing the experiment. Gin configs are described in detail in a [later section](#gin-configs).
- `config` - Gin config overrides, applied _after_ the config file. There can be many of them.
- `output_dir` - Experiment output directory. Checkpoints and the operative Gin config are saved there.

## Core abstractions

### Runner

[`Runner`](alpacka/runner.py) is the main class of the experiment, taking care of running the training loop. Each iteration consists of two phases: gathering data from environments and training networks.

### Agent & Trainer

The logic of an RL algorithm is split into two classes: [`Agent`](alpacka/agents/base.py) collecting experience by trying to solve the environment and [`Trainer`](alpacka/trainers/base.py) training the neural networks on the collected data.

<!--TODO(xxx): Describe Agent/OnlineAgent, Trainer and their responsibilities.-->

<!--TODO(xxx): DeterministicMCTS. -->

### Network

[`Network/TrainableNetwork`](alpacka/networks/core.py) abstracts out the deep learning framework used for network inference and training.

### BatchStepper

[`BatchStepper`](alpacka/batch_steppers.py) is an abstraction for the parallelization strategy of data collection. It takes care of running a batch of episodes with a given `Agent` on a given `Env` using a given `Network` for inference. We currently support 2 strategies:

- Local execution - `LocalBatchStepper`: Single node, `Network` inference batched across `Agent` instances. Running the environments is sequential. This setup is good for running light `Env`s with heavy `Network`s that benefit a lot from hardware acceleration, on a single node with or without GPU.
- Distributed execution using [Ray](https://ray.readthedocs.io/en/latest/) - `RayBatchStepper`: Multiple workers on single or multiple nodes, every worker has its own `Agent`, `Env` and `Network`. This setup is good for running heavy `Env`s and scales well with the number of nodes.

## Important concepts

### Gin configs

We use [Gin](https://github.com/google/gin-config) to manage experiment configurations. The assumption is that all the "conventional" experiments supported by Alpacka can be run using the entrypoint `alpacka/runner.py` with an appropriate Gin config specified so that in most cases you won't need to write your own training pipeline, but rather override the necessary classes and provide your own Gin config. If you have a use-case that warrants customizing the entrypoint, please contact us - it might make sense to support it natively in Alpacka.

Some predefined config files are in `configs`. Some are usage examples for the implemented algorithms, and some serve as regression tests for the most important experiments.

<!-- TODO(xxx): How they're structured (top-down config writing process), where to get them from (operative configs). -->

### Coroutines

We make heavy use of Python coroutines for communication between Agents and Networks. This allows us to abstract away the paralleization strategy, while retaining a simple API for Agents using Networks internally:

```python
class ExampleAgent(Agent):

    def solve(self, env, **kwargs):
        # Planning...
        predictions = yield inputs
        # Planning...
        predictions = yield inputs
        # Planning...
        return episode
```

where `inputs` signify an input to the network, and `predictions` are the predicted output.

The most common use of coroutines in Python is to implement asynchronous computation, e.g. using the awesome [`asyncio`](https://docs.python.org/3/library/asyncio.html) library. This is **not** what Alpacka uses coroutines for. We only import `asyncio` for the `@asyncio.coroutine` decorator, marking a function as a coroutine even if it doesn't have a `yield` statement inside.

For a quick summary of coroutines and their use in `asyncio`, please refer to <http://masnun.com/2015/11/13/python-generators-coroutines-native-coroutines-and-async-await.html>.

### Pytrees

Both inputs and outputs to a network can be arbitrary nested structures of Python tuples, namedtuples, lists and dicts with numpy arrays at leaves. We call such nested data structures *pytrees*. This design allows the networks to operate on arbitrarily complicated data structures and to build generic data structures, such as [replay buffers](alpacka/trainers/replay_buffers.py), abstracting away the specific data layout. We define various utility functions operating on pytrees in [alpacka.data](alpacka/data/ops.py). Their usecases are best illustrated in the [tests](alpacka/data/ops_test.py).

The concept of pytrees comes from [JAX](https://github.com/google/jax), a next-generation numerical computation and deep learning library. Pytrees emulate algebraic datatypes from functional programming languages: products => tuples, coproducts (union types) => different types of namedtuples.
