import unittest

from app.infra.singleton import Singleton


class TestSingleton(unittest.TestCase):
    def test_singleton_instance_is_single(self):
        instance1 = Singleton()
        instance2 = Singleton()

        self.assertIs(
            instance1,
            instance2,
            "Singleton instances should be identical (same object)",
        )

    def test_singleton_instance_shares_state(self):
        instance1 = Singleton()
        instance2 = Singleton()

        # Add attribute via one instance
        instance1.some_value = 42

        # Check it's visible from the other instance
        self.assertEqual(instance2.some_value, 42)

    def test_singleton_only_created_once(self):
        # Reset for this test
        Singleton._instance = None

        class MySingleton(Singleton):
            def __init__(self):
                if not hasattr(self, "initialized"):
                    self.initialized = True
                    self.created = True
                else:
                    self.created = False

        a = MySingleton()
        self.assertTrue(a.created)

        b = MySingleton()

        self.assertIs(a, b)
        self.assertTrue(hasattr(a, "initialized"))
        self.assertFalse(a.created)
        self.assertFalse(b.created)  # Second instance should not re-init


if __name__ == "__main__":
    unittest.main()
