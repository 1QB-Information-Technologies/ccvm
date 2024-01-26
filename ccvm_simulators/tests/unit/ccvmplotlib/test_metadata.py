import unittest
import json
import os
import shutil
from ccvm_simulators.solvers.ccvm_solver import DeviceType
from ccvm_simulators.metadata import Metadata


class TestMetadataClass(unittest.TestCase):
    def setUp(self):
        # Create a Metadata instance for testing
        self.test_dir = "./test_metadata"
        self.device = DeviceType.CPU_DEVICE.value
        self.metadata = Metadata(device=self.device)

    def tearDown(self):
        # Clean up: remove the temporary directory and its contents
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_to_result_metadata(self):
        result_dict = {"problem_size": 20, "batch_size": 1000}
        self.metadata.add_to_result_metadata(result_dict)
        self.assertEqual(self.metadata.result_metadata, [result_dict])

    def test_finalize_metadata(self):
        # Initially, metadata_dict should be None
        self.assertIsNone(self.metadata.metadata_dict)

        # After calling finalize_metadata, metadata_dict should have the expected structure
        self.metadata.finalize_metadata()
        self.assertIsNotNone(self.metadata.metadata_dict)
        self.assertEqual(self.metadata.metadata_dict["device"], self.device)
        self.assertListEqual(self.metadata.metadata_dict["result_metadata"], [])

    def test_save_metadata_to_file(self):
        test_dir = self.test_dir

        # Check the test directory does not exist
        self.assertFalse(os.path.exists(test_dir))

        # Test saving metadata to a file
        metadata_file_path = self.metadata.save_metadata_to_file(
            file_dir=test_dir, file_name="test_metadata"
        )

        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.exists(metadata_file_path))

        # Check if the saved file contains valid JSON
        with open(metadata_file_path, "r") as file:
            saved_metadata = json.load(file)
            self.assertEqual(saved_metadata["device"], self.device)

            # Check if the key "result_metadata" exists
            self.assertIn("result_metadata", saved_metadata)


if __name__ == "__main__":
    unittest.main()
