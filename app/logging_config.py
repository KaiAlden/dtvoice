"""统一日志格式，避免业务模块各自配置日志处理器。"""

from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    """仅在未配置根日志时建立控制台日志。"""

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
