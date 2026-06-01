import os
import sys
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile

from config import VicRegConfig as Config


def fast_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not os.path.exists(dst) or os.path.getsize(src) != os.path.getsize(dst):
        shutil.copy2(src, dst)


if __name__ == "__main__":
    cfg = Config()

    if cfg.TRAIN_LOCATION == "local":
        raise ValueError("Staging script should only be run on cluster. Exiting.")

    global_base = cfg.GLOBAL_DATA_DIR
    local_base = cfg.LOCAL_DATA_DIR

    dataset = cfg.DATASET

    copy_tasks = []
    match dataset:
        case "iNaturalist":
            fast_copy(os.path.join(global_base, "train_mini.tar.gz"), os.path.join(local_base, ""))
            fast_copy(os.path.join(global_base, "val.tar.gz"), os.path.join(local_base, ""))

            with tarfile.open(os.path.join(local_base, "train_mini.tar.gz"), "r:gz") as tar:
                tar.extractall()
            with tarfile.open(os.path.join(local_base, "val.tar.gz"), "r:gz") as tar:
                tar.extractall()
        case _:
            raise ValueError("Dataset not supported for staging. Exiting.")