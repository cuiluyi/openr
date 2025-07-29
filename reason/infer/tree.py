"""
The Node and MCTS class for AlphaZero.
"""

import math
import heapq
import pprint

from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from loguru import logger
from reason.env import Env
from reason.node import Node, LanguageNode
from utils.distributed import print_rank_0, print_with_rank


class SearchTree:
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20)

        # UCB formula
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)  # 19652
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            "root_dirichlet_alpha", 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)  # 0.25

        self.root = None

        self.answers = set()
        self.wrong_answers = set()
        self.visited_paths = None

        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        self.mask_non_terminal_node_value = self._cfg.get("mask_non_terminal_node_value", False)

        self._completion_tokens = 0

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def vanila_mcts(
        self,
        simulate_env: Type[Env],
        num_path: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        traj_list = []

        # TODO(ziyu): split with 1. select 2. expand 3. rollout 4. backprop
        #  for here is split the for loop with select and rollout
        #  so that arbitrary rollout function can be used here.

        for i_path in range(num_path):
            node = self.root
            env_copy = simulate_env.copy()
            done = False
            while not done:
                if node.visit_count > 0:
                    # if node is visited, select the child with the highest UCB score
                    action, node = self._select_child(node, env_copy)
                else:
                    # select with highest value, since visit_count = 0 in self.ucb
                    #  will select node with highest value
                    action, node = self._select_child(node, env_copy)

                # sync terminated flag here
                # XXX(ziyu): find a more clean way
                env_copy._next_state_terminated = {}
                assert node.last_action == action
                env_copy._next_state_terminated[action] = node.terminated

                _, _, terminated, truncated, info = env_copy.step(
                    action,
                    value=node._initial_value,
                    update_legal_action=node.is_leaf(),
                )

                done = terminated or truncated

                if not done and node.is_leaf():
                    self._expand_leaf_node(node, env_copy, reward_model_fn)

                # record api_tokens, if not expand, info["api_completion_token"] is 0
                api_call_completion_tokens += info["api_completion_token"]
            else:
                if node.visit_count > 0:
                    leaf_value = node.value
                else:
                    leaf_value = node._initial_value
            node.update_recursive(leaf_value, env_copy.mcts_mode)

            traj_data = {
                "path_idx": i_path,
                "text": env_copy.answer,
                "values": env_copy.values,
                "api_completion_tokens": api_call_completion_tokens,
                "tree_completion_tokens": self._completion_tokens,
            }

            traj_list.append(traj_data)

            # reset api_call_completion_tokens
            api_call_completion_tokens = 0

        return traj_list

    def beam_search(
        self,
        simulate_env: Env,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """Beam Search implementation
        Args:
            simulate_env: The environment to simulate the search.
            beam_size: beam_size
            max_step: The maximum number of steps to search.
            reward_model_fn: The reward model function to evaluate the state.
        """
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        end_nodes, top_k_nodes = [], [(-root._initial_value, root, simulate_env.copy())]
        k = beam_size

        for _ in range(max_step + 1):
            cur_nodes_to_search = top_k_nodes
            top_k_nodes = []
            for cur_neg_v, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:
                    end_nodes.append((cur_neg_v, cur_node, cur_env))
                    k -= 1
                elif k > 0:
                    # select at most topk children add push to heap
                    assert (
                        len(cur_node.children) > 0
                    ), "in beam search you should expand this non-terminal node at first."

                    top_k_children = sorted(
                        [
                            (action, child, child._initial_value)
                            for action, child in cur_node.children.items()
                        ],
                        key=lambda x: x[2],
                        reverse=True,
                    )[:k]
                    for _, c_node, c_value in top_k_children:
                        new_env = cur_env.copy()
                        heapq.heappush(top_k_nodes, (-c_value, c_node, new_env))

            # nsmallest since we negate the value
            top_k_nodes = heapq.nsmallest(k, top_k_nodes)

            # expand selected nodes
            # XXX(ziyu): this could be optimized by batch expand
            for value, node, new_env in top_k_nodes:
                _, _, terminated, truncated, info = new_env.step(
                    node.last_action,
                    node._initial_value,
                    update_legal_action=True,
                )
                api_call_completion_tokens += info["api_completion_token"]
                if terminated or truncated:
                    node.set_as_terminate_node()
                else:
                    self._expand_leaf_node(node, new_env, reward_model_fn)

            if len(end_nodes) == beam_size:
                assert k == 0
                break

        traj_list = []
        for i, (neg_e_v, e_node, e_env) in enumerate(end_nodes):
            traj_list.append(
                {
                    "path_idx": i,
                    "text": e_env.answer,
                    "values": e_env.values,
                    "api_completion_tokens": 0,
                    "tree_completion_tokens": 0,
                }
            )
        traj_list[-1]["tree_completion_tokens"] = self._completion_tokens
        traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens
        return traj_list

    def _select_child(
        self,
        node: LanguageNode,
        criteria: str = "puct",
    ) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - criteria (:obj:`Str`): The criteria to select the child node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """

        action = None
        child = None
        best_score = -9999999

        for action_tmp, child_tmp in node.children.items():
            if criteria == "ucb":
                score = self._ucb_score(node, child_tmp)
            elif criteria == "uct":
                score = self._uct_score(node, child_tmp)
            elif criteria == "puct":
                score = self._puct_score(node, child_tmp)
            elif criteria == "visit_count":
                score = child_tmp.visit_count
            else:
                score = child_tmp.value

            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: Type[Env],
        reward_fn: Optional[Callable] = None,
    ) -> None:
        """
        Overview:
            expand the node with the reward_fn.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - reward_fn (:obj:`Function`): the Callable to compute the state value.
        """

        text_state = simulate_env.get_state()

        assert len(simulate_env.legal_actions) > 0

        prms: List[List[float]] = reward_fn(
            [
                (
                    simulate_env.question,
                    simulate_env.answer + x["action"],
                )
                for x in simulate_env.legal_actions
            ]
        )

        # PRM get last r as single reward
        child_values = []
        for act, rs in zip(simulate_env.legal_actions, prms):
            if len(simulate_env.action_history) + 1 != len(rs):
                log_message = (
                    "PRM value length not match with action history.\n" + "=" * 80 + "\n"
                    f"PRM Length:        {len(prms)}\n"
                    f"ActionHist Length: {len(simulate_env.action_history)}\n\n"
                    + "-" * 28
                    + " State "
                    + "-" * 43
                    + "\n"
                    f"{pprint.pformat(text_state)}\n" + "-" * 28 + " Action " + "-" * 42 + "\n"
                    f"{pprint.pformat(act)}\n" + "-" * 28 + " Rewards " + "-" * 41 + "\n"
                    f"{pprint.pformat(rs)}\n" + "=" * 80
                )
                logger.warning(log_message)
                child_values.append(0.0)
            elif len(rs) == 0:
                logger.warning(
                    f"Empty PRM value for: \nState: \n{text_state} \naction: \n{act}, will be set to 0.0"
                )
                child_values.append(0.0)
            else:
                child_values.append(rs[-1])  # prm-last
                # child_values.append(min(rs))  # prm-min
                # child_values.append(act["prob"])  # prob-prm

        assert len(node.children) == 0
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]

            child_value = child_values[i]

            node.children[action] = LanguageNode(
                parent=node,
                prior_p=prob,
                text_state=text_state,
                last_action=action,
                initial_value=child_value,
                num_generated_token=action_dict["num_token"],
            )
            # set terminal node here
            if simulate_env._next_state_terminated[action]:
                node.children[action].set_as_terminate_node()
        if len(node.children) == 0:
            print_rank_0("Prune all current children at node {}".format(node.last_action))

        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(c.num_generated_token for c in node.children.values())
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

    # TODO: has problems
    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        value_score = child.value

        return value_score + math.sqrt(self.root.visit_count / (child.visit_count + 0.000001))

    def _puct_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute PUCT score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """

        c_puct = math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        value_score = child.value
        prior_score = c_puct * child.prior_p * math.sqrt(parent.visit_count) / (1 + child.visit_count)

        return value_score + prior_score

    # TODO: has problems
    def _uct_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCT score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCT score.
        """
        value_score = child.value

        return value_score + self._pb_c_init * math.sqrt(math.log(parent.visit_count)) / (
            child.visit_count + 0.000001
        )
