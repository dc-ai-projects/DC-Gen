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

import os
import queue
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import ray
import torch
from omegaconf import MISSING
from tqdm import tqdm

from .config import get_config


class SubtaskStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MultiProcessConfig:
    use_ray: bool = True
    num_workers: int = 16
    run_dir: str = MISSING
    num_subtasks_per_task: Optional[int] = None
    task_id: Optional[int] = None
    remaining_subtask_ids: Optional[tuple[int]] = None
    occupy_device: bool = False
    ignore_error: bool = True


@dataclass
class ChunkingMultiProcessConfig(MultiProcessConfig):
    """
    Assign a contiguous subtask chunk to each worker for better locality
    """

    chunk_size: int = 1000


class Distributor:
    """
    A `task` is done by a `Distributor` and multiple `Workers` and can be divided into several `subtasks`.
    A `subtask` is done by a `Worker` each time and is non-divisible.
    """

    def __init__(self, cfg: MultiProcessConfig):
        self.cfg = cfg
        os.makedirs(cfg.run_dir, exist_ok=True)

        if cfg.occupy_device:
            self.occupy_device()

        self.subtasks = self.build_subtasks()
        self.subtask_ids_iterator = iter(self.subtasks)
        if cfg.task_id is None:
            self.db_path = os.path.join(cfg.run_dir, "subtasks.db")
        else:
            self.db_path = os.path.join(cfg.run_dir, f"subtasks_{cfg.task_id}.db")
        self.init_db()
        self.completed_subtask_ids = self.get_completed_subtask_ids()
        self.worker_status_list: list[tuple[Optional[int], datetime]] = [
            (None, datetime.now()) for _ in range(cfg.num_workers)
        ]
        self.tqdm = tqdm(
            total=len(self.subtasks),
            desc="working",
            initial=len(self.completed_subtask_ids),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            file=sys.stdout,
        )

    def occupy_device(self):
        self.occupant_list = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.occupant_list.append(torch.randn((10 * 1024**3 // 4,), device=torch.device(f"cuda:{i}")))

    def build_all_subtasks(self) -> dict[int, Any]:
        raise NotImplementedError

    def build_subtasks(self) -> dict[int, Any]:
        subtasks = self.build_all_subtasks()
        print(f"num subtasks: {len(subtasks)}")
        if self.cfg.remaining_subtask_ids is not None:
            subtasks = {
                subtask_id: subtask
                for subtask_id, subtask in subtasks.items()
                if subtask_id in self.cfg.remaining_subtask_ids
            }
        if self.cfg.num_subtasks_per_task is not None and self.cfg.task_id is not None:
            num_subtasks = len(subtasks)
            start_id = min(self.cfg.task_id * self.cfg.num_subtasks_per_task, num_subtasks)
            end_id = min(start_id + self.cfg.num_subtasks_per_task, num_subtasks)
            subtasks = dict(sorted(subtasks.items())[start_id:end_id])
        return subtasks

    def init_db(self):
        db_exists = os.path.exists(self.db_path)
        db_path = self.db_path if db_exists else self.db_path + ".tmp"
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subtasks (
                    subtask_id INTEGER PRIMARY KEY,
                    status TEXT DEFAULT 'started',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    worker_id INTEGER
                )
            """
            )
            conn.commit()
        # avoid partially created database file
        if not db_exists:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA journal_mode=DELETE;")  # merge -shm and -wal
            os.rename(db_path, self.db_path)

    @contextmanager
    def get_db_connection(self, db_path: Optional[str] = None):
        """Context manager for database connections"""
        conn = sqlite3.connect(db_path or self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_completed_subtask_ids(self) -> set[int]:
        """Get all completed subtask ids"""
        completed_subtask_ids = set()
        for file_name in os.listdir(self.cfg.run_dir):
            if not file_name.endswith(".db"):
                continue
            with self.get_db_connection(os.path.join(self.cfg.run_dir, file_name)) as conn:
                table_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='subtasks'"
                ).fetchone()
                if table_exists:
                    cursor = conn.execute("SELECT subtask_id FROM subtasks WHERE status='completed'")
                    completed_subtask_ids.update(set(row[0] for row in cursor.fetchall()))
        return completed_subtask_ids & set(self.subtasks.keys())

    def mark_finished(self, worker_id: int, subtask_id: int, start_time: datetime, subtask_status: SubtaskStatus):
        """Mark subtask as completed"""
        if subtask_status != SubtaskStatus.COMPLETED:
            return
        with self.get_db_connection() as conn:
            conn.execute(
                "INSERT INTO subtasks (subtask_id, status, started_at, completed_at, worker_id) VALUES (?, ?, ?, ?, ?)",
                (
                    subtask_id,
                    subtask_status.value,
                    start_time,
                    datetime.now(),
                    worker_id,
                ),
            )

    def get_next_subtask_id(self, worker_id: int) -> Optional[int]:
        while True:
            try:
                subtask_id = next(self.subtask_ids_iterator)
            except StopIteration:
                return None
            if subtask_id not in self.completed_subtask_ids:
                return subtask_id

    def get_next_subtask(self, worker_id: int, current_subtask_status: SubtaskStatus) -> Optional[Any]:
        current_subtask_id, start_time = self.worker_status_list[worker_id]
        if current_subtask_id is not None:
            self.mark_finished(worker_id, current_subtask_id, start_time, current_subtask_status)
            self.tqdm.update()
        next_subtask_id = self.get_next_subtask_id(worker_id)
        if next_subtask_id is None:
            return None
        self.worker_status_list[worker_id] = (next_subtask_id, datetime.now())
        return self.subtasks[next_subtask_id]


class ChunkingDistributor(Distributor):
    """
    Assign a contiguous subtask chunk to each worker for better locality
    """

    def __init__(self, cfg: ChunkingMultiProcessConfig):
        super().__init__(cfg)
        self.cfg: ChunkingMultiProcessConfig
        self.assigned_subtasks_queue_list: list[queue.Queue[int]] = [
            queue.Queue(maxsize=cfg.chunk_size) for _ in range(cfg.num_workers)
        ]

    def get_next_subtask_id(self, worker_id: int) -> Optional[int]:
        assigned_subtasks_queue = self.assigned_subtasks_queue_list[worker_id]
        if assigned_subtasks_queue.empty():
            for _ in range(self.cfg.chunk_size):
                try:
                    subtask_id = next(self.subtask_ids_iterator)
                    assigned_subtasks_queue.put(subtask_id)
                except StopIteration:
                    break
        if assigned_subtasks_queue.empty():
            return None
        else:
            return assigned_subtasks_queue.get()


class Worker:
    def __init__(self, cfg: MultiProcessConfig, distributor: Distributor, worker_id: int):
        self.cfg = cfg
        self.distributor = distributor
        self.worker_id = worker_id

    def do_subtask(self, subtask) -> SubtaskStatus:
        raise NotImplementedError

    def work(self):
        last_subtask_status = SubtaskStatus.FAILED
        while True:
            if self.cfg.use_ray:
                subtask = ray.get(self.distributor.get_next_subtask.remote(self.worker_id, last_subtask_status))
            else:
                subtask = self.distributor.get_next_subtask(self.worker_id, last_subtask_status)
            if subtask is None:
                break
            if self.cfg.ignore_error:
                try:
                    last_subtask_status = self.do_subtask(subtask)
                except Exception as e:
                    print(f"subtask {subtask} failed with error {e}")
                    last_subtask_status = SubtaskStatus.FAILED
            else:
                last_subtask_status = self.do_subtask(subtask)


# example
def main():
    cfg = get_config(MultiProcessConfig)
    if cfg.use_ray:
        ray.init(ignore_reinit_error=True, num_cpus=5 * cfg.num_workers + 1)

        @ray.remote(num_cpus=1)
        class DistributorRemote(Distributor):
            pass

        @ray.remote(num_cpus=1)
        class WorkerRemote(Worker):
            pass

        distributor = DistributorRemote.remote(cfg)
        workers = [WorkerRemote.remote(cfg, distributor, worker_id) for worker_id in range(cfg.num_workers)]
        ray.get([worker.work.remote() for worker in workers])
    else:
        distributor = Distributor(cfg)
        worker = Worker(cfg, distributor, 0)
        worker.work()
