import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("\033[94miterative-label-refinement\033[0m %(levelname)s (%(asctime)s)\t%(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def set_log_level(level: str) -> None:
    match level.lower():
        case "debug":
            logger.setLevel(logging.DEBUG)
        case "info":
            logger.setLevel(logging.INFO)
        case "warning":
            logger.setLevel(logging.WARNING)
        case "error":
            logger.setLevel(logging.ERROR)
        case "critical":
            logger.setLevel(logging.CRITICAL)
        case _:
            raise ValueError(f"Unknown log level {level}")
