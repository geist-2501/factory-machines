import unittest

import numpy as np

from agents.slam_astar import SlamAstar


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


if __name__ == '__main__':
    unittest.main()
