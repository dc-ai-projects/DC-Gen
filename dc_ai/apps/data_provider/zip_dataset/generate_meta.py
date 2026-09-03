# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import zipfile
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

from omegaconf import MISSING


def generate_and_load_zip_meta(zip_path: str, cache_dir: str, overwrite: bool = False) -> dict[str, Any] | None:
    """
    Generates and saves metadata for a single .zip file.

    The metadata includes the number of unique samples (based on filenames
    without extensions), the total filesize, and the file's path (url).
    The result is cached in a JSON file to avoid re-computation.

    Args:
        zip_path: The absolute path to the .zip file.
        cache_dir: The directory where the metadata JSON file will be cached.
        overwrite: If True, forces regeneration of the metadata even if a cached version exists.

    Returns:
        A dictionary containing the metadata for the zip file, or None if an error occurs.
    """
    # Create a unique, filesystem-safe name for the cache file.
    zip_meta_path = os.path.join(
        os.path.expanduser(cache_dir),
        zip_path.replace(os.path.sep, "--") + ".json",
    )

    if not os.path.exists(zip_meta_path) or overwrite:
        print(f"Generating meta for: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Get unique sample keys by taking the filename without the extension.
                # This correctly groups files like 'sample_01.jpg' and 'sample_01.json'
                # under a single sample key 'sample_01'.
                keys = {os.path.splitext(name)[0] for name in zf.namelist() if not name.endswith("/")}
                nsamples = len(keys)
        except (zipfile.BadZipFile, FileNotFoundError) as e:
            print(f"Skipping unloadable or missing zip file: {zip_path}\nError: {e}")
            return None

        zip_meta = {
            "url": zip_path,
            "nsamples": nsamples,
            "filesize": os.path.getsize(zip_path),
        }
        os.makedirs(os.path.dirname(zip_meta_path), exist_ok=True)
        with open(zip_meta_path, "w") as f:
            json.dump(zip_meta, f, indent=4)

        return zip_meta

    # print(f"Loading meta from cache: {zip_meta_path}")
    with open(zip_meta_path, "r") as f:
        return json.load(f)


def generate_meta(
    data_dir: str,
    cache_dir: str = "~/.cache/zip_dataset_meta",
    save_path: str | None = None,
    processes: int = 10,
) -> None:
    """
    Crawls a directory to find all .zip files and generates a consolidated metadata file.

    Args:
        data_dir: The root directory of the dataset containing .zip shards.
        cache_dir: A directory to cache metadata for individual shards.
        save_path: The path to save the final consolidated 'wids-meta.json' file.
                   Defaults to a file in the data_dir.
        processes: The number of worker processes to use for generating metadata in parallel.
    """
    data_dir = os.path.expanduser(data_dir)
    cache_dir = os.path.expanduser(cache_dir)

    # 1. Find all .zip files in the data directory
    zip_path_list = []
    for root, _, file_names in os.walk(data_dir):
        for file_name in file_names:
            if file_name.endswith(".zip"):
                file_path = os.path.join(root, file_name)
                zip_path_list.append(file_path)

    zip_path_list.sort()

    if not zip_path_list:
        print(f"Warning: No .zip files were found in {data_dir}")
        return

    print(f"Found {len(zip_path_list)} zip files. Generating metadata...")

    # 2. Process each zip file in parallel
    with Pool(processes=processes) as pool:
        args_list = [(zip_path, cache_dir) for zip_path in zip_path_list]
        # Use starmap to pass multiple arguments to the worker function
        zip_meta_list = pool.starmap(generate_and_load_zip_meta, args_list)

    # Filter out any shards that failed to process (returned None)
    valid_meta_list = [meta for meta in zip_meta_list if meta is not None]

    if not valid_meta_list:
        print("Error: Metadata generation failed for all zip files.")
        return

    # 3. Create the final metadata dictionary
    meta = {
        "wids_version": 1,
        # Sort the shardlist by URL for consistent output
        "shardlist": sorted(valid_meta_list, key=lambda x: x["url"]),
    }

    # 4. Save the consolidated metadata file
    if save_path is None:
        save_path = os.path.join(data_dir, "wids-meta.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Successfully generated metadata for {len(valid_meta_list)} shards.")
    print(f"Consolidated metadata saved to: {save_path}")


@dataclass
class GenerateMetaConfig:
    data_dir: str = MISSING
    save_path: str | None = None
    test_load_data: bool = False


def main():
    from omegaconf import OmegaConf

    cfg: GenerateMetaConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(GenerateMetaConfig), OmegaConf.from_cli())
    )
    generate_meta(cfg.data_dir, save_path=cfg.save_path)


if __name__ == "__main__":
    main()
