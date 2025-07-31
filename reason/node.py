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
        initial_value: float = 0.0,
    ) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self._initial_value = initial_value
        self._terminated = False
        self._prob = prob

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._initial_value
        return self._value_sum / self._visit_count

    @property
    def prob(self):
        return self._prob

    @property
    def terminated(self):
        return self._terminated

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    def set_as_terminate_node(self):
        self._terminated = True

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

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
        self._parent.update_recursive(leaf_value)


class LanguageNode(Node):
    state: Optional[str] = None
    action: Optional[str] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prob: float = 1.0,
        state: Optional[str] = None,
        action: Optional[str] = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
    ) -> None:
        super().__init__(parent, prob, initial_value)
        self.state = state
        self.action = action

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

    def get_path(self) -> str:
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_root(self) -> "LanguageNode":
        node = self
        while not node.is_root():
            node = node.parent
        return node
    
    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.state)
        else:
            return "action: {}, value: {:.3f}, prob: {:.3f}".format(self.action, self.value, self._prob)
