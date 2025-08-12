from typing import List, Dict, Optional, Callable, Type

from reason.env import Env


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(
        self,
        parent: "Node" = None,
        prob: float = 1.0,
        state: Optional[str] = "",
        action: Optional[str] = "",
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = 0,
    ) -> None:
        self.parent = parent
        self.prob = prob
        self.state = state
        self.action = action
        self.initial_value = initial_value
        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.terminated = False

    def __lt__(self, other):
        return self.initial_value < other.initial_value

    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.state)
        else:
            return "action: {}, value: {:.3f}, prob: {:.3f}".format(
                self.action, self.value, self.prob
            )

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        # Option 1: return a weighted average of the initial value and the current value
        # EPSILON = 0.5
        # return (
        #     EPSILON * self.initial_value + (1 - EPSILON) * self.value_sum / self.visit_count
        #     if self.visit_count > 0
        #     else self.initial_value
        # )
        # # Option 2: if not visited, return 0
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

        # # Option 3: if not visited, return the initial value: PRM(s1, s2, ..., sn)
        # if self.visit_count == 0:
        #     return self.initial_value
        # return self.value_sum / self.visit_count

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self.children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self.parent is None

    def get_path(self) -> str:
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.action)
            node = node.parent
        return "".join(reversed(ans))

    def get_values(self) -> List[float]:
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.value)
            node = node.parent
        return ans.reverse()

    def get_root(self) -> "Node":
        node = self
        while not node.is_root():
            node = node.parent
        return node

    def set_as_terminate_node(self):
        self.terminated = True

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self.visit_count += 1
        self.value_sum += value

    def update_recursive(self, leaf_value: float) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        self.update(leaf_value)
        if self.is_root():
            return
        self.parent.update_recursive(leaf_value)
