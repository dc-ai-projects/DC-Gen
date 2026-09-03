import json
import os
import tarfile
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

from omegaconf import MISSING

from .wids import WebDataset


def generate_and_load_tar_meta(tar_path: str, cache_dir: str, overwrite: bool = False) -> dict:
    tar_meta_path = os.path.join(
        os.path.expanduser(cache_dir),
        tar_path.lstrip("/") + ".json",
    )

    if not os.path.exists(tar_meta_path) or overwrite:
        print(f"Generating meta: {tar_meta_path}")
        try:
            tar = tarfile.open(tar_path)
            uuids = set([os.path.splitext(_)[0] for _ in tar.getnames()])
            if "." in uuids:
                uuids.remove(".")  # for sam
        except tarfile.ReadError as e:
            print(f"Skipping {tar_path}")
            print(e)
            return None
        nsamples = len(uuids)

        tar_meta = {
            "url": tar_path,
            "nsamples": nsamples,
            "filesize": os.path.getsize(tar_path),
        }
        os.makedirs(os.path.dirname(tar_meta_path), exist_ok=True)
        json.dump(tar_meta, open(tar_meta_path, "w"), indent=4)

    print(f"Loading abs meta: {tar_meta_path}")
    tar_meta = json.load(open(tar_meta_path, "r"))
    return tar_meta


def generate_meta(
    data_dir: str,
    cache_dir: str = "~/.cache/web_dataset_meta",
    save_path: Optional[str] = None,
    overwrite: bool = False,
    processes: int = 10,
) -> None:
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    cache_dir = os.path.expanduser(cache_dir)
    tar_path_list = []
    for root, _, file_names in os.walk(data_dir):
        for file_name in file_names:
            if not file_name.endswith(".tar"):
                continue
            file_path = os.path.join(root, file_name)
            tar_path_list.append(file_path)
    tar_path_list = sorted(tar_path_list)

    assert len(tar_path_list) > 0, f"no tar was found in the repository {data_dir} !"
    print(f"generating meta for total {len(tar_path_list)} files.")

    with Pool(processes=processes) as pool:
        args_list = [(tar_path, cache_dir, overwrite) for tar_path in tar_path_list]
        tar_meta_list = pool.starmap(generate_and_load_tar_meta, args_list)

    if save_path is None:
        save_path = os.path.join(data_dir, "wids-meta.json")
    else:
        save_path = os.path.abspath(save_path)

    for tar_meta in tar_meta_list:
        tar_meta["url"] = os.path.relpath(tar_meta["url"], save_path)

    meta = {
        "wids_version": 1,
        "path_format": "relative",
        "base_dir": os.path.relpath(data_dir, save_path),
        "shardlist": sorted(tar_meta_list, key=lambda x: x["url"]),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(meta, open(save_path, "w"), indent=4)


@dataclass
class GenerateMetaConfig:
    data_dir: str = MISSING
    save_path: Optional[str] = None
    test_load_data: bool = False
    overwrite: bool = False


def main():
    from omegaconf import OmegaConf

    cfg: GenerateMetaConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(GenerateMetaConfig), OmegaConf.from_cli())
    )
    generate_meta(cfg.data_dir, save_path=cfg.save_path, overwrite=cfg.overwrite)

    if cfg.test_load_data:
        dataset = WebDataset(cfg.data_dir, cfg.save_path)
        print(f"dataset size: {len(dataset)}")
        print(dataset[0])

        # data_loader = torch.utils.data.DataLoader(
        #     dataset,
        #     shuffle=False,
        #     batch_size=8,
        #     num_workers=8,
        # )
        # for idx, data in tqdm(enumerate(data_loader)):
        #     pass


if __name__ == "__main__":
    main()
