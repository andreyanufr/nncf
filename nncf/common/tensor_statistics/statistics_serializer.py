# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, cast

import nncf
from nncf.common.tensor_statistics.statistics_validator import validate_cache
from nncf.common.utils.backend import BackendType
from nncf.common.utils.os import safe_open
from nncf.tensor import functions as fns
from nncf.tensor.tensor import Tensor
from nncf.tensor.tensor import get_tensor_backend

METADATA_FILE = "statistics_metadata.json"
STATISTICS_FILE_EXTENSION = ".safetensors"


def sanitize_filename(filename: str) -> str:
    """
    Replaces any forbidden characters with an underscore.

    :param filename: Original filename.
    :return: Sanitized filename with no forbidden characters.
    """
    return re.sub(r"[^\w]", "_", filename)


def add_unique_name(name: str, unique_map: Dict[str, List[str]]) -> str:
    """
    Creates an unique name, adds it to a `unique_map` and returns this unique name.

    :param name: The original name.
    :param unique_map: A dictionary mapping names to lists of unique sanitized names.
    :return: A unique name generated by appending a count to the original name.
    """
    # Next number of the same sanitized name
    count = len(unique_map[name]) + 1
    unique_sanitized_name = f"{name}_{count}"
    unique_map[name].append(unique_sanitized_name)
    return unique_sanitized_name


def load_metadata(dir_path: Path) -> Dict[str, Any]:
    """
    Loads the metadata, including the mapping and any other metadata information from the metadata file.

    :param dir_path: The directory where the metadata file is stored.
    :return: A dictionary containing the metadata.
    """
    metadata_file = dir_path / METADATA_FILE
    if metadata_file.exists():
        with safe_open(metadata_file, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    raise nncf.StatisticsCacheError(f"Metadata file does not exist in the following path: {dir_path}")


def save_metadata(metadata: Dict[str, Any], dir_path: Path) -> None:
    """
    Saves metadata to a file in the specified directory.

    :param metadata: Dictionary containing metadata and mapping.
    :param dir_path: Path to the directory where the metadata file will be saved.
    """
    metadata_file = dir_path / METADATA_FILE
    with safe_open(metadata_file, "w") as f:
        json.dump(metadata, cast(TextIO, f), indent=4)


def load_statistics(dir_path: Path, backend: BackendType) -> Dict[str, Dict[str, Tensor]]:
    """
    Loads statistics from a directory.

    :param dir_path: The path to the directory from which to load the statistics.
    :param backend: Backend type to determine the tensor backend.
    :return: Statistics.
    """
    metadata = load_metadata(dir_path)
    try:
        validate_cache(metadata, dir_path, backend)
        statistics: Dict[str, Dict[str, Tensor]] = {}
        mapping = metadata.get("mapping", {})
        tensor_backend = get_tensor_backend(backend)
        for file_name, original_name in mapping.items():
            statistics_file = dir_path / file_name
            statistics[original_name] = fns.io.load_file(statistics_file, backend=tensor_backend)  # no device support
        return statistics
    except Exception as e:
        raise nncf.StatisticsCacheError(str(e))


def dump_statistics(
    statistics: Dict[str, Dict[str, Tensor]],
    dir_path: Path,
    backend: BackendType,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves statistics and metadata to a directory.

    Metadata is stored in a JSON file named "statistics_metadata.json".
    Statistics are stored in individual files with sanitized and unique filenames to prevent collisions.

    Metadata Format:
    The metadata file must have a mapping of saved filenames to the original names and backend type.
    {
        "mapping": {
            "saved_file_name_1": "original_name_1",
            "saved_file_name_2": "original_name_2",
            ...
        },
        "backend": "backend_type",
        ... (additional metadata fields)
    }

    :param statistics: A dictionary with statistic names as keys and the statistic data as values.
    :param dir_path: The path to the directory where the statistics will be dumped.
    :param backend: Backend type to save in metadata.
    :param additional_metadata: A dictionary containing any additional metadata to be saved with the mapping.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {"mapping": {}, "backend": backend.value}
    unique_map: Dict[str, List[str]] = defaultdict(list)
    for original_name, statistics_value in statistics.items():
        sanitized_name = sanitize_filename(original_name)
        unique_sanitized_name = add_unique_name(sanitized_name, unique_map) + STATISTICS_FILE_EXTENSION

        file_path = dir_path / unique_sanitized_name

        # Update the mapping
        metadata["mapping"][unique_sanitized_name] = original_name

        try:
            fns.io.save_file(statistics_value, file_path)
        except Exception as e:
            raise nncf.InternalError(f"Failed to write data to file {file_path}: {e}")

    if additional_metadata:
        metadata |= additional_metadata

    save_metadata(metadata, dir_path)
