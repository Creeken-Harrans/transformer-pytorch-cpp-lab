"""小矩阵测试用单线程，避免并行调度掩盖教学模型本身的运行成本。"""

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)
