import logging


def set_logger(name: str) -> logging.Logger:
    """set logger"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        log_hdl = logging.StreamHandler()
        log_hdl.setLevel(logging.DEBUG)
        log_fmt = logging.Formatter(
            fmt=('%(asctime)s.%(msecs)03d, %(levelname)s, %(name)s,'
                 ' %(module)s, 999999, %(pathname)s:%(lineno)d,'
                 ' "%(message)s"'),
            datefmt="%Y-%m-%dT%H:%M:%S")
        log_hdl.setFormatter(log_fmt)
        logger.addHandler(log_hdl)

    # capture python warning
    logger_pywarn = logging.getLogger('py.warnings')
    for hdl in logger.handlers:
        logger_pywarn.addHandler(hdl)
    logging.captureWarnings(True)

    return logger
