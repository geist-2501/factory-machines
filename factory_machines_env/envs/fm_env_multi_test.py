import unittest

import numpy as np

from factory_machines_env.envs import FactoryMachinesEnvMulti


class FMFlattenRelativeTest(unittest.TestCase):
    def should_terminate(self):
        # TODO
        env = FactoryMachinesEnvMulti(
            map_id="1",
            order_override=[

            ]
        )
        pass

    def should_get_correct_depot_ages(self):
        # TODO
        pass

    def should_clear_agent_inventory_correctly(self):
        # TODO
        pass
