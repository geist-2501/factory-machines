import unittest

from talos.error import ProfilePropertyNotFound
from talos.profile import read_profile


class ProfileTest(unittest.TestCase):
    def test_should_load_profile(self):
        profiles = read_profile("test_profiles.yaml")

        self.assertTrue("final_dqn_map_0" in profiles)

        profile = profiles["final_dqn_map_0"]
        self.assertEqual(profile.env_id, "FMMulti-v0")

    def test_should_override_base_config(self):
        profile = read_profile("test_profiles.yaml")["final_dqn_map_0"]
        self.assertEqual(profile.config.getint("init_epsilon"), 1)
        self.assertEqual(profile.config.getint("batch_size"), 128)

    def test_should_get_list(self):
        profile = read_profile("test_profiles.yaml")["final_dqn_map_0"]
        hidden_layers = profile.config.getlist("hidden_layers")
        self.assertTrue(type(hidden_layers) is list)
        self.assertTrue(type(hidden_layers[0]) is int)

    def test_should_get_float_in_exponent_notation(self):
        profile = read_profile("test_profiles.yaml")["final_dqn_map_0"]
        self.assertAlmostEqual(profile.config.getfloat("learning_rate"), 1e-4)

    def test_should_get_non_existent(self):
        profile = read_profile("test_profiles.yaml")["final_dqn_map_0"]
        self.assertEqual(profile.config.getint("beep boop", required=False), None)
        self.assertRaises(ProfilePropertyNotFound, lambda: profile.config.getint("beep boop"))


if __name__ == '__main__':
    unittest.main()