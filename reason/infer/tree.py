"""
The Node and MCTS class for AlphaZero.
"""

import math
import heapq
import pprint
import jsonlines

from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from loguru import logger
from reason.env import Env
from reason.node import Node
from utils.distributed import print_rank_0, print_with_rank


class SearchTree:
    """
    Overview:
        MCTS search process.
    """

    def __init__(
        self,
        pb_c_base: int = 19652,
        pb_c_init: float = 1.25,
    ) -> None:
        # UCB formula
        self._pb_c_base = pb_c_base
        self._pb_c_init = pb_c_init
        self.root = None
        self._completion_tokens = 0

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def vanilla_mcts(
        self,
        simulate_env: Env,
        num_path: int,
        rm_call: Optional[Callable] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        api_call_completion_tokens = 0
        api_completion_token = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += api_completion_token
        if self.root is None:
            root = Node(state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, rm_call)
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
                action, node = self._select_child(node)  # PUCT default

                # XXX(ziyu): find a more clean way
                env_copy.next_state_terminated = {}
                assert node.action == action
                env_copy.next_state_terminated[action] = node.terminated

                api_completion_token = env_copy.step(
                    action,
                    value=node.initial_value,
                    update_legal_action=node.is_leaf(),
                )
                api_call_completion_tokens += api_completion_token

                done = env_copy.reason_finished
                if not done and node.is_leaf():
                    self._expand_leaf_node(node, env_copy, rm_call)

            if node.visit_count > 0:
                leaf_value = node.value
            else:
                leaf_value = node.initial_value
            node.update_recursive(leaf_value)

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

        # collect step data
        tree_step_data = self.dfs_non_leaf_nodes(self.root)

        return traj_list, tree_step_data

    def beam_search(
        self,
        simulate_env: Env,
        beam_size: int,
        max_step: int,
        rm_call: Optional[Callable] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Beam Search implementation
        Args:
            simulate_env: The environment to simulate the search.
            beam_size: beam_size
            max_step: The maximum number of steps to search.
            rm_call: The reward model function to evaluate the state.
        """
        api_call_completion_tokens = 0
        api_completion_token = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += api_completion_token
        if self.root is None:
            root = Node(state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, rm_call)
            self.root = root

        endnode_envs, top_k_nodes = [], [(-root.initial_value, root, simulate_env.copy())]
        k = beam_size

        for _ in range(max_step + 1):
            cur_nodes_to_search = top_k_nodes
            top_k_nodes = []
            for _, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:
                    endnode_envs.append(cur_env)
                    k -= 1
                elif k > 0:
                    # select at most topk children add push to heap
                    assert (
                        len(cur_node.children) > 0
                    ), "in beam search you should expand this non-terminal node at first."

                    top_k_children = sorted(
                        [(action, child, child.initial_value) for action, child in cur_node.children.items()],
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
            for _, node, new_env in top_k_nodes:
                api_completion_token = new_env.step(
                    node.action,
                    node.initial_value,
                    update_legal_action=True,
                )
                api_call_completion_tokens += api_completion_token

                done = new_env.reason_finished
                if done:
                    node.set_as_terminate_node()
                else:
                    self._expand_leaf_node(node, new_env, rm_call)

            if len(endnode_envs) == beam_size:
                assert k == 0
                break

        traj_list = []
        for i, endnode_env in enumerate(endnode_envs):
            traj_list.append(
                {
                    "path_idx": i,
                    "text": endnode_env.answer,
                    "values": endnode_env.values,
                    "api_completion_tokens": 0,
                    "tree_completion_tokens": 0,
                }
            )
        traj_list[-1]["tree_completion_tokens"] = self._completion_tokens
        traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens

        # collect step data
        tree_step_data = self.dfs_non_leaf_nodes(self.root)

        return traj_list, tree_step_data

    def dfs_non_leaf_nodes(
        self,
        node: Node,
    ) -> List[Dict]:
        results = []

        def dfs(node: Node):
            if node.is_leaf():
                return

            prompt = node.state + node.action

            for child in node.children.values():
                results.append(
                    {
                        "prompt": prompt,
                        "completion": child.action,
                        "prob": child.prob,
                        "num_tokens": child.num_generated_token,
                        "value": child.value,
                    }
                )
                dfs(child)

        dfs(node)
        return results

    def _select_child(
        self,
        node: Node,
        criteria: str = "puct",
    ) -> Tuple[Optional[str], Node]:
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

        select_child_step = None
        select_child_node = None
        max_score = -9999999

        for child_step, child_node in node.children.items():
            if criteria == "ucb":
                score = self._ucb_score(node, child_node)
            elif criteria == "uct":
                score = self._uct_score(node, child_node)
            elif criteria == "puct":
                score = self._puct_score(node, child_node)
            elif criteria == "visit_count":
                score = child_node.visit_count
            else:
                score = child_node.value

            if score > max_score:
                max_score = score
                select_child_step = child_step
                select_child_node = child_node

        # child==None, node is leaf node
        if select_child_node is None:
            select_child_node = node

        return select_child_step, select_child_node

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: Env,
        rm_call: Optional[Callable] = None,
    ) -> None:
        """
        Overview:
            expand the node with the rm_call.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - rm_call (:obj:`Function`): the Callable to compute the state value.
        """

        state = simulate_env.get_state()

        assert len(simulate_env.legal_actions) > 0

        prms: List[List[float]] = rm_call(
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
                    f"{pprint.pformat(state)}\n" + "-" * 28 + " Action " + "-" * 42 + "\n"
                    f"{pprint.pformat(act)}\n" + "-" * 28 + " Rewards " + "-" * 41 + "\n"
                    f"{pprint.pformat(rs)}\n" + "=" * 80
                )
                logger.warning(log_message)
                child_values.append(0.0)
            elif len(rs) == 0:
                logger.warning(
                    f"Empty PRM value for: \nState: \n{state} \naction: \n{act}, will be set to 0.0"
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

            node.children[action] = Node(
                parent=node,
                prob=prob,
                state=state,
                action=action,
                initial_value=child_value,
                num_generated_token=action_dict["num_token"],
            )
            # set terminal node here
            if simulate_env.next_state_terminated[action]:
                node.children[action].set_as_terminate_node()
        if len(node.children) == 0:
            print_rank_0("Prune all current children at node {}".format(node.action))

        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(
                child_node.num_generated_token for child_node in node.children.values()
            )
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
        prior_score = c_puct * child.prob * math.sqrt(parent.visit_count) / (1 + child.visit_count)

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
