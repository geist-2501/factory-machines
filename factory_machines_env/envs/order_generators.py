from abc import ABC, abstractmethod
from typing import TypeVar, List

import numpy as np
from scipy.stats import norm

OrderType = TypeVar("OrderType")


class OrderGenerator(ABC):
    """Abstract base class for generating an order."""
    @abstractmethod
    def should_make_order(self, num_current_orders: int) -> bool:
        """Check whether a new order should be made."""
        raise NotImplementedError

    @abstractmethod
    def make_order(self, size) -> OrderType:
        """Make an order."""
        raise NotImplementedError


class BinomialOrderGenerator(OrderGenerator):
    """Order generator that that has a binomial probability for each item."""

    def should_make_order(self, num_current_orders: int) -> bool:
        return bool(np.random.binomial(1, 0.1)) or num_current_orders == 0

    def make_order(self, size) -> OrderType:
        order = np.zeros(size, dtype=int)
        while sum(order) == 0:
            order = (np.random.binomial(1, 0.5, size=size)).astype(int)

        return order


class GaussianOrderGenerator(OrderGenerator):

    def __init__(self, p: List[float], max_order_size: int) -> None:
        super().__init__()
        p = np.array(p)
        self._p = p / p.sum()
        self._max = max_order_size

    def should_make_order(self, num_current_orders: int) -> bool:
        return bool(np.random.binomial(1, 0.1)) or num_current_orders == 0

    def make_order(self, size) -> OrderType:
        order = np.zeros(size, dtype=int)
        num_items = np.floor(np.random.triangular(1, self._max / 2, self._max)).astype(int)
        items = np.random.choice(size, size=num_items, p=self._p, replace=False)
        order[items] = 1
        return order


class MockOrderGenerator(OrderGenerator):
    """A fake order generator for testing!"""

    def __init__(self, orders: List) -> None:
        super().__init__()
        self._orders = orders

    def should_make_order(self, num_current_orders: int) -> bool:
        if len(self._orders) == 0:
            return False

        pending_order = self._orders[0]
        if pending_order is None:
            self._orders.pop(0)
            return False
        else:
            return True

    def make_order(self, size) -> OrderType:
        assert len(self._orders) != 0
        order = self._orders.pop(0)
        assert order is not None
        assert len(order) == size
        return order
