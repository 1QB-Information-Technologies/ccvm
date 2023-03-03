import json
import torch
import os
from dataclasses import dataclass, field, asdict


class MetadataList:
    """Define the metadata list class."""

    def __init__(self):
        """The constructor for MetadataList."""
        # Empty list to store metadata
        self.metadata_list = []

    def add_metadata(self, metadata):
        """Add metadata dict to the metadata list.

        Args:
            metadata (dict): The metadata dict to be added to the metadata list.
        """
        self.metadata_list.append(metadata)

    def save_metadata_to_file(self, file_dir="./metadata", file_name="metadata"):
        """Save the metadata to the defined file.

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

        try:
            with open(metadata_file_path, "w") as outfile:
                json.dump(self.metadata_list, outfile)
                print(
                    f"Successfully saved metadata to metadata_file_path: {metadata_file_path}"
                )
                return metadata_file_path
        except Exception as e:
            raise Exception("Error saving metadata to file: " + str(e))
