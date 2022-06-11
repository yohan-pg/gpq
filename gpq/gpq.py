import nvsmi
import time
import os
import psutil
import torch
from typing import Optional, List, Tuple
import persistqueue
from datetime import datetime
import longprocess


PID = int


class PIDQueue:
    Item = Tuple[datetime, PID]

    def _get_queue(self):
        return persistqueue.Queue("/tmp/easy_queue")

    def peek(self) -> PID:
        queue = self._get_queue()

        while not psutil.pid_exists(pid := queue.get()[1]):  # type: ignore
            queue.task_done()

        return pid  # type: ignore

    def list(self) -> List[Item]:
        queue = self._get_queue()
        items = []

        try:
            while item := queue.get(block=False):
                items.append(item)
        except persistqueue.Empty:
            pass

        return items

    def pop(self) -> None:
        queue = self._get_queue()
        queue.get()
        queue.task_done()

    def enqueue_self(self):
        self._get_queue().put((datetime.now(), pid := os.getpid()))
        print(f"Enqueued {pid}")


class GPU:
    def __init__(self, gpu_spec: nvsmi.GPU):
        self.gpu_spec = gpu_spec

    @staticmethod
    def priority(gpu_spec: nvsmi.GPU) -> int:
        return -gpu_spec.mem_free

    @staticmethod
    def get_next_free() -> Optional["GPU"]:
        if gpu_list := sorted(nvsmi.get_available_gpus(), key=GPU.priority):
            return GPU(gpu_list[0])
        else:
            return None

    def select(self) -> None:
        torch.cuda.set_device(int(self.gpu_spec.id))
        # Allocate some dummy value, forcing initialization and marking the gpu as utilized
        torch.empty(1).cuda()

        print(f"Started on gpu {self.gpu_spec.id}")


queue = PIDQueue()


def wait_for_turn(polling_interval_secs: int = 2) -> None:
    longprocess.linger()

    queue.enqueue_self()
    had_to_wait_yet = False

    while True:
        if queue.peek() == os.getpid() and (gpu := GPU.get_next_free()):
            gpu.select()
            queue.pop()
            return
        elif not had_to_wait_yet:
            had_to_wait_yet = True
            print("Waiting for turn...")
        time.sleep(polling_interval_secs)


def print_full_queue() -> None:
    for timestamp, pid in queue.list():
        if psutil.pid_exists(int(pid)):
            print(timestamp.isoformat(timespec="seconds") + " " + str(pid))


if __name__ == "__main__":
    print_full_queue()
