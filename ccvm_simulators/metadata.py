import json
import os


class Metadata:
    """Define the metadata class."""

    def __init__(self, device):
        """The constructor for metadata."""
        self.device = device
        self.result_metadata = []
        self.metadata_dict = None

    def add_to_result_metadata(self, result_dict):
        """Add result dict to the result metadata list.

        Args:
            result_dict (dict): The result dict to be added to the result metadata list.
        """
        self.result_metadata.append(result_dict)

    def finalize_metadata(self):
        """Finalize the metadata list."""

        self.metadata_dict = {
            "device": self.device,
            "result_metadata": self.result_metadata,
        }

    def save_metadata_to_file(self, file_dir="./metadata", file_name="metadata"):
        """Save the metadata dict to the defined file.

        Args:
            file_dir (str, optional): The file directory where the
            metadata file will be stored. Defaults to "./metadata".
            file_name (str, optional): The name of file containing the metadata. Defaults to "metadata".

        Raises:
            Exception: Failed to create folder.
            Exception: Failed to save metadata to file.

        Returns:
            str: File path of the metadata file
        """
        # If file_dir not exists, create the path
        try:
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
                print("The folder to store doesn't exist yet. Creating: ", file_dir)
        except Exception as e:
            raise Exception(f"Failed to create the folder path: {e}")

        metadata_file_path = f"{file_dir}/{file_name}.json"

        if self.metadata_dict is None:
            self.finalize_metadata()

        try:
            with open(metadata_file_path, "w") as outfile:
                json.dump(self.metadata_dict, outfile)
                print(
                    f"Successfully saved metadata to metadata_file_path: {metadata_file_path}"
                )
                return metadata_file_path
        except Exception as e:
            raise Exception("Error saving metadata to file: " + str(e))
