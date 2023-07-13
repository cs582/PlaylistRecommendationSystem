import unittest
from src.unit_tests.transformer_tests import ClassifierUnitTest


if __name__=="__main__":
    # Recommender test
    recommender_test = unittest.TestLoader().loadTestsFromTestCase(ClassifierUnitTest)

    # List all tests
    tests = [recommender_test]

    # Test suite that includes all the tests
    suite = unittest.TestSuite(tests)

    # Run the test suite
    unittest.TextTestRunner().run(suite)
