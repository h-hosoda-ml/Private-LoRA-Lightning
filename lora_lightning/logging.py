import os
import logging

# Python logger
logger = logging.getLogger("lora_lightning")


def setup_logging(log_dir: str | None = None, reset_log: bool = True):
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if log_dir:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_path = os.path.join(log_dir, "log.txt")

        if reset_log:
            with open(log_file_path, "w") as f_log:
                pass

        # TODO: fileHandlerの存在確認
        file_handler_exists = any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        )
        if not file_handler_exists:
            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"logは '{log_file_path}' に出力されます。")
