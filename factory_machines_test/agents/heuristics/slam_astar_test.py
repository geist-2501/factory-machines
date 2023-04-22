import unittest

import numpy as np

from factory_machines.agents.heuristics.slam_astar import SlamAstar


class SlamAstarTest(unittest.TestCase):

    def test_should_initialise_correctly(self):
        initial_obs = np.zeros((3, 3))
        slam = SlamAstar(initial_obs)

        self.assertListEqual(slam.map.tolist(), initial_obs.tolist())

    def test_should_expand_bottom_right(self):
        initial_obs = np.zeros((3, 3))
        slam = SlamAstar(initial_obs)

        new_obs = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [0, 1, 0]
        ])
        new_pos = np.array([1, 2])
        slam.update(new_pos, new_obs)

        len_y, len_x = slam.map.shape
        self.assertEqual(len_y, 5)
        self.assertEqual(len_x, 4)

        expected_map = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
        ])

        self.assertListEqual(slam.map.tolist(), expected_map.tolist())

    def test_should_expand_top_left(self):
        initial_obs = np.zeros((3, 3))
        slam = SlamAstar(initial_obs)

        new_obs = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        new_pos = np.array([-1, -2])
        slam.update(new_pos, new_obs)

        len_y, len_x = slam.map.shape
        self.assertEqual(len_y, 5)
        self.assertEqual(len_x, 4)

        expected_map = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        self.assertListEqual(slam.map.tolist(), expected_map.tolist())

    def test_should_calc_simple_path(self):
        blank_obs = np.zeros((3, 3))
        slam = SlamAstar(blank_obs)
        slam.update(np.array([2, 2]), blank_obs)

        path = slam.astar(np.array([0, 0]), np.array([3, 2]))

        self.assertIsNotNone(path)
        self.assertEqual(len(path), 6)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[5], (3, 2))

    def test_should_calc_path_around_wall(self):
        initial_obs = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])
        slam = SlamAstar(initial_obs)
        slam.update(np.array([2, 0]), np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]))

        path = slam.astar(np.array([0, 0]), np.array([2, -1]))

        self.assertIsNotNone(path)
        self.assertEqual(len(path), 6)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[5], (2, -1))

    def test_should_calc_alternate_path(self):
        initial_obs = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ])
        slam = SlamAstar(initial_obs)
        slam.update(np.array([1, 0]), np.array([
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]))

        path = slam.get_path(np.array([0, 0]), np.array([4, 0]))

        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
        self.assertEqual(path[2], (1, 1))

if __name__ == '__main__':
    unittest.main()
