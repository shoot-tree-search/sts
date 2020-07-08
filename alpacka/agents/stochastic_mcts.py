"""Monte Carlo Tree Search for stochastic environments."""

import math

import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.agents import core
from alpacka.utils import space as space_utils


def _uniform_prior(n):
    return np.full(shape=(n,), fill_value=(1 / n))


class NewLeafRater:
    """Base class for rating the children of an expanded leaf."""

    def __init__(self, discount):
        """Initializes NewLeafRater.

        Args:
            discount (float): Discount factor.
        """
        self._discount = discount

    def __call__(self, observation, model):
        """Rates the children of an expanded leaf.

        Args:
            observation (np.ndarray): Observation received at leaf.
            model (gym.Env): Model environment.

        Yields:
            Network prediction requests.

        Returns:
            list: List of pairs (quality, prob) for each action, where quality
                is the estimated quality of the action and prob is its prior
                probability (e.g. from a policy network).
        """
        raise NotImplementedError

    def network_signature(self, observation_space, action_space):
        """Defines the signature of networks used by this NewLeafRater.

        Args:
            observation_space (gym.Space): Environment observation space.
            action_space (gym.Space): Environment action space.

        Returns:
            NetworkSignature or None: Either the network signature or None if
            the NewLeafRater doesn't use a network.
        """
        raise NotImplementedError


@gin.configurable
class RolloutNewLeafRater(NewLeafRater):
    """Rates new leaves using rollouts with an Agent."""

    def __init__(
        self,
        discount,
        rollout_agent_class=core.RandomAgent,
        rollout_time_limit=100,
    ):
        super().__init__(discount)
        self._agent = rollout_agent_class()
        self._time_limit = rollout_time_limit

    def __call__(self, observation, model):
        init_state = model.clone_state()

        child_qualities = []
        for init_action in space_utils.element_iter(model.action_space):
            (observation, init_reward, done, _) = model.step(init_action)
            yield from self._agent.reset(model, observation)
            value = 0
            total_discount = 1
            time = 0
            while not done and time < self._time_limit:
                (action, _) = yield from self._agent.act(observation)
                (observation, reward, done, _) = model.step(action)
                value += total_discount * reward
                total_discount *= self._discount
                time += 1
            child_qualities.append(init_reward + self._discount * value)
            model.restore_state(init_state)
        prior = _uniform_prior(len(child_qualities))
        return list(zip(child_qualities, prior))

    def network_signature(self, observation_space, action_space):
        return self._agent.network_signature(observation_space, action_space)


@gin.configurable
class ValueNetworkNewLeafRater(NewLeafRater):
    """Rates new leaves using a value network."""

    def __init__(self, discount, use_policy=False):
        super().__init__(discount)
        self._use_policy = use_policy

    def __call__(self, observation, model):
        init_state = model.clone_state()

        def step_and_rewind(action):
            (observation, reward, done, _) = model.step(action)
            model.restore_state(init_state)
            return (observation, reward, done)

        (observations, rewards, dones) = data.nested_stack([
            step_and_rewind(action)
            for action in space_utils.element_iter(model.action_space)
        ])
        # TODO(xxx): We currently do two network predictions here. It's
        # possible to do just one - rewrite.
        # Run the network to predict values for children.
        if not self._use_policy:
            values = yield observations
        else:
            (values, _) = yield observations
        # (batch_size, 1) -> (batch_size,)
        values = np.reshape(values, -1)
        # Compute the final qualities, masking out the "done" states.
        child_qualities = list(rewards + self._discount * values * (1 - dones))
        if not self._use_policy:
            prior = _uniform_prior(len(child_qualities))
        else:
            (_, prior) = yield observation[None, ...]
            prior = prior[0]
        return list(zip(child_qualities, prior))

    def network_signature(self, observation_space, action_space):
        n_actions = space_utils.max_size(action_space)
        if self._use_policy:
            return data.NetworkSignature(
                input=space_utils.signature(observation_space),
                output=(
                    data.TensorSignature(shape=(1,)),
                    data.TensorSignature(shape=(n_actions,))
                ),
            )
        else:
            # Input: observation, output: scalar value.
            return data.NetworkSignature(
                input=space_utils.signature(observation_space),
                output=data.TensorSignature(shape=(1,)),
            )


@gin.configurable
class QualityNetworkNewLeafRater(NewLeafRater):
    """Rates new leaves using a value network."""

    def __init__(self, discount, use_policy=False):
        super().__init__(discount)
        self._use_policy = use_policy

    def __call__(self, observation, model):
        del model
        result = yield np.expand_dims(observation, axis=0)
        if self._use_policy:
            (qualities, prior) = result
            qualities = qualities[0]
            prior = prior[0]
        else:
            qualities = result[0]
            prior = _uniform_prior(qualities.shape[0])
        return list(zip(qualities, prior))

    def network_signature(self, observation_space, action_space):
        n_actions = space_utils.max_size(action_space)
        action_vector_sig = data.TensorSignature(shape=(n_actions,))
        if self._use_policy:
            output_sig = (action_vector_sig,) * 2
        else:
            output_sig = action_vector_sig
        # Input: observation, output: quality vector and optionally policy
        # vector.
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=output_sig,
        )


@gin.configurable
def puct_exploration_bonus(child_count, parent_count):
    """PUCT exploration bonus.

    A variant with weight changing over time is used in AlphaZero.

    Args:
        child_count (int): Number of visits in the child node so far.
        parent_count (int): Number of visits in the parent node so far.

    Returns:
        float: Exploration bonus to apply to the child.
    """
    return math.sqrt(parent_count) / (child_count + 1)


class TreeNode:
    """Node of the search tree.

    Attrs:
        children (list): List of children, indexed by action.
        is_leaf (bool): Whether the node is a leaf, i.e. has not been expanded
            yet.
    """

    def __init__(self, init_quality=None, prior_probability=None):
        """Initializes TreeNode.

        Args:
            init_quality (float or None): Quality received from
                the NewLeafRater for this node, or None if it's the root.
            prior_probability (float): Prior probability of picking this node
                from its parent.
        """
        self._quality_sum = 0
        self._quality_count = 0
        if init_quality is not None:
            self.visit(init_quality)
        self.prior_probability = prior_probability
        self.children = []

    def visit(self, quality):
        """Records a visit in the node during backpropagation.

        Args:
            quality (float): Quality accumulated on the path out of the
                node.
        """
        self._quality_sum += quality
        self._quality_count += 1

    @property
    def quality(self):
        """Returns the quality of going into this node in the search tree.

        We use it instead of value, so we can handle dense rewards.
        quality(s, a) = reward(s, a) + discount * value(s')
        """
        return self._quality_sum / self._quality_count

    @property
    def count(self):
        return self._quality_count

    @property
    def value(self):
        """Returns the value of going into this node in the search tree.

        We use it only to provide targets for value network training.
        value(s) = expected_a quality(s, a)
        """
        if self.is_leaf:
            return 0
        else:
            return (
                sum(child.quality * child.count for child in self.children) /
                sum(child.count for child in self.children)
            )

    @property
    def is_leaf(self):
        return not self.children


class StochasticMCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search for stochastic environments."""

    def __init__(
        self,
        n_passes=10,
        discount=0.99,
        new_leaf_rater_class=RolloutNewLeafRater,
        exploration_bonus_fn=puct_exploration_bonus,
        exploration_weight=1.0,
        action_selection_mode='count',
        sampling_temperature=0.0,
        depth_limit=float('+inf'),
        prior_noise_weight=0.0,
        prior_noise_parameter=1.0,
        n_leaves_to_expand=1,
        **kwargs
    ):
        """Initializes StochasticMCTSAgent.

        Args:
            n_passes (int): Number of MCTS passes per act().
            discount (float): Discount factor.
            new_leaf_rater_class (type): NewLeafRater for estimating qualities
                of new leaves.
            exploration_bonus_fn (callable): Function calculating an
                exploration bonus for a given node. It's added to the node's
                quality when choosing a node to explore in an MCTS pass.
                Signature: (child_count, parent_count) -> bonus.
            exploration_weight (float): Weight of the exploration bonus.
            action_selection_mode (str): Mode of selecting the final action.
                Either 'count' - select based on the action visit count, or
                'quality' - select based on the estimated quality.
            sampling_temperature (float): Sampling temperature for choosing the
                actions on the real environment.
            depth_limit (int): Maximum number of nodes in a single pass.
            prior_noise_weight (float): Weight of the Dirichlet noise added to
                the prior.
            prior_noise_parameter (float): Parameter of the Dirichlet noise
                added to the prior.
            n_leaves_to_expand (int): Number of consecutive leaves to expand in
                a single MCTS pass.
            kwargs: OnlineAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self.n_passes = n_passes
        self._discount = discount
        self._new_leaf_rater = new_leaf_rater_class(self._discount)
        self._exploration_bonus = exploration_bonus_fn
        self._exploration_weight = exploration_weight
        self._action_selection_mode = action_selection_mode
        self._sampling_temperature = sampling_temperature
        self._depth_limit = depth_limit
        self._prior_noise_weight = prior_noise_weight
        self._prior_noise_parameter = prior_noise_parameter
        self._n_leaves_to_expand = n_leaves_to_expand
        self._model = None
        self._root = None
        self._root_state = None

    def _choose_action(self, node, exploratory):
        """Chooses the action to take in a given node based on child qualities.

        Args:
            node (TreeNode): Node to choose an action from.
            exploratory (bool): Whether the choice should be exploratory (in
                an MCTS pass) or not (when choosing the final action on the real
                environment).

        Returns:
            Action to take.
        """
        def rate_child(child):
            if exploratory:
                quality = child.quality + (
                    self._exploration_weight * child.prior_probability *
                    self._exploration_bonus(child.count, node.count)
                )
            else:
                if self._action_selection_mode == 'count':
                    quality = np.log(child.count)
                elif self._action_selection_mode == 'quality':
                    quality = child.quality
                else:
                    raise ValueError(
                        'action_selection_mode ' + self._action_selection_mode +
                        ' is not supported.'
                    )
                # Sample an action to perform on the real environment using
                # Gumbel sampling. No need to normalize logits.
                u = np.random.uniform(low=1e-6, high=1.0 - 1e-6)
                g = -np.log(-np.log(u))
                quality += g * self._sampling_temperature
            return quality

        child_qualities = [rate_child(child) for child in node.children]
        child_qualities_and_actions = zip(
            child_qualities, range(len(child_qualities))
        )

        (_, action) = max(child_qualities_and_actions)
        return action

    def _traverse(self, root, observation):
        """Chooses a path from the root to a leaf in the search tree.

        Does not modify the nodes.

        Args:
            root (TreeNode): Root of the search tree.
            observation (np.ndarray): Observation received at root.

        Yields:
            Network prediction requests.

        Returns:
            Tuple (path, observation, done), where path is a list of pairs
            (reward, node) of nodes visited during traversal and rewards
            collected when stepping into them, observation is the observation
            received in the leaf, done is the "done" flag received when stepping
            into the leaf. In case of a "done", traversal is interrupted.
        """
        path = [(0, root)]
        node = root
        done = False
        n_leaves_left = self._n_leaves_to_expand
        quality = 0.
        while not node.is_leaf and not done and len(path) < self._depth_limit:
            agent_info = self._compute_node_info(node)

            action = self._choose_action(node, exploratory=True)
            node = node.children[action]
            (observation, reward, done, _) = self._model.step(action)

            for callback in self._callbacks:
                callback.on_model_step(
                    agent_info, action, observation, reward, done
                )

            path.append((reward, node))

            if node.is_leaf and n_leaves_left > 0:
                if not done:
                    quality = yield from self._expand_leaf(node, observation)
                n_leaves_left -= 1

        if node.is_leaf and not done and n_leaves_left > 0:
            # Corner case: we've reached the depth limit and we're at the leaf.
            # We need to expand it.
            quality = yield from self._expand_leaf(node, observation)

        return (path, quality)

    def _expand_leaf(self, leaf, observation):
        """Expands a leaf and returns its quality.

        The leaf's new children are assigned initial quality. The quality of the
        "best" new leaf is then backpropagated.

        Only modifies leaf - adds children with new qualities.

        Args:
            leaf (TreeNode): Leaf to expand.
            observation (np.ndarray): Observation received at leaf.

        Yields:
            Network prediction requests.

        Returns:
            float: Quality of a chosen child of the expanded leaf.
        """
        child_qualities_and_probs = yield from self._new_leaf_rater(
            observation, self._model
        )
        # This doesn't work with dynamic action spaces. TODO(xxx): Fix.
        assert len(child_qualities_and_probs) == space_utils.max_size(
            self._action_space
        )
        leaf.children = [
            TreeNode(quality, prob)
            for (quality, prob) in child_qualities_and_probs
        ]
        if leaf is self._root:
            self._add_prior_noise(leaf)
        action = self._choose_action(leaf, exploratory=True)
        return leaf.children[action].quality

    def _backpropagate(self, quality, path):
        """Backpropagates quality to the root through path.

        Only modifies the qualities of nodes on the path.

        Args:
            quality (float): Quality collected at the leaf.
            path (list): List of (reward, node) pairs, describing a path from
                the root to a leaf.
        """
        for (reward, node) in reversed(path):
            quality = reward + self._discount * quality
            node.visit(quality)

    def _run_pass(self, root, observation):
        """Runs a pass of MCTS.

        A pass consists of:
            1. Tree traversal to find a leaf.
            2. Expansion of the leaf, adding its successor states to the tree
               and rating them.
            3. Backpropagation of the value of the best child of the old leaf.

        During leaf expansion, new children are rated only using
        the NewLeafRater - no actual stepping into those states in the
        environment takes place for efficiency, so that NewLeafRater can
        be implemented by running a neural network that rates all children of
        a given node at the same time. In case of a "done", traversal is
        interrupted, the leaf is not expanded and value 0 is backpropagated.

        Args:
            root (TreeNode): Root node.
            observation (np.ndarray): Observation collected at the root.

        Yields:
            Network prediction requests.
        """
        for callback in self._callbacks:
            callback.on_pass_begin()

        (path, quality) = yield from self._traverse(root, observation)
        self._backpropagate(quality, path)
        # Go back to the root state.
        self._model.restore_state(self._root_state)

        for callback in self._callbacks:
            callback.on_pass_end()

    def reset(self, env, observation):
        """Reinitializes the search tree for a new environment."""
        yield from super().reset(env, observation)
        self._model = env
        self._root = TreeNode()

    def act(self, observation):
        """Runs n_passes MCTS passes and chooses the best action."""
        assert self._model is not None, (
            'MCTSAgent works only in model-based mode.'
        )
        self._root_state = self._model.clone_state()
        for _ in range(self.n_passes):
            yield from self._run_pass(self._root, observation)
        info = {'_node': self._root}
        info.update(self._compute_node_info(self._root))
        info.update(self._compute_tree_metrics(self._root))

        action = self._choose_action(self._root, exploratory=False)
        self._root = self._root.children[action]
        if self._root.children:
            self._add_prior_noise(self._root)
        return (action, info)

    def _add_prior_noise(self, node):
        prior = np.array([child.prior_probability for child in node.children])
        noise = np.random.dirichlet(
            [self._prior_noise_parameter] *
            space_utils.max_size(self._action_space)
        )
        prior = (
            (1 - self._prior_noise_weight) * prior +
            self._prior_noise_weight * noise
        )
        for (child, p) in zip(node.children, prior):
            child.prior_probability = p

    def postprocess_transitions(self, transitions):
        for transition in transitions:
            transition.agent_info.update(
                self._compute_node_info(transition.agent_info.pop('_node'))
            )
        return transitions

    def _compute_node_info(self, node):
        value = node.value
        qualities = np.array([child.quality for child in node.children])
        action_counts = np.array([child.count for child in node.children])
        # "Smooth" histogram takes into account the initial actions
        # performed on all children of an expanded leaf, resulting in
        # a more spread out distribution.
        action_histogram_smooth = action_counts / np.sum(action_counts)
        # Ordinary histogram only takes into account the actual actions
        # chosen in the inner nodes.
        action_histogram = (action_counts - 1) / np.sum(action_counts - 1)
        prior_probabilities = np.array([
            child.prior_probability for child in node.children
        ])
        exploration_bonuses = self._exploration_weight * np.array([
            self._exploration_bonus(child.count, node.count)
            for child in node.children
        ])
        total_scores = qualities + prior_probabilities * exploration_bonuses
        return {
            'value': value,
            'qualities': qualities,
            'action_histogram_smooth': action_histogram_smooth,
            'action_histogram': action_histogram,
            'prior_probabilities': prior_probabilities,
            'exploration_bonuses': exploration_bonuses,
            'total_scores': total_scores,
        }

    @staticmethod
    def compute_metrics(episodes):
        def episode_info(key):
            for episode in episodes:
                yield from episode.transition_batch.agent_info[key]

        def entropy(probs):
            def plogp(p):
                # If out this case to avoid log(0).
                return p * np.log(p) if p else 0
            return -np.sum([plogp(p) for p in probs])

        return {
            'depth_mean': np.mean(list(episode_info('depth_mean'))),
            'depth_max': max(episode_info('depth_max')),
            'entropy_mean': np.mean(
                list(map(entropy, episode_info('action_histogram')))
            ),
            'entropy_smooth_mean': np.mean(
                list(map(entropy, episode_info('action_histogram_smooth')))
            ),
        }

    def _compute_tree_metrics(self, root):
        # TODO: unit-test this function
        leaf_depths = list()
        path = [root]
        # number of visited children for given node in path
        children_visited = [0]

        def go_to_parent():
            path.pop()
            children_visited.pop()

        # Iterate over leaves with DFS
        while path:
            node = path[-1]
            if node.is_leaf:
                leaf_depths.append(len(path) - 1)
                go_to_parent()
            elif children_visited[-1] == len(node.children):
                # all children of given node were already visited
                go_to_parent()
            else:
                # expand new child
                path.append(node.children[children_visited[-1]])
                children_visited[-1] += 1
                children_visited.append(0)

        return {
            'depth_mean': sum(leaf_depths) / len(leaf_depths),
            'depth_max': max(leaf_depths),
        }

    def network_signature(self, observation_space, action_space):
        # Delegate defining the network signature to NewLeafRater. This is the
        # only part of the agent that uses a network, so it should decide what
        # sort of network it needs.
        return self._new_leaf_rater.network_signature(
            observation_space, action_space
        )
