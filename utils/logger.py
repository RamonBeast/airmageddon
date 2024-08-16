from enum import Enum
from datetime import datetime

class LogLevel(Enum):
    Notification = ('notification', '[\033[1;32m\N{check mark}\033[0m]')
    Error = ('error', '[\033[1;31m\N{aegean check mark}\033[0m]')
    Info = ('info', '[\033[1;34m\N{information source}\033[0m]')
    Warning = ('warning', '[\033[1;35m\N{warning sign}\033[0m]')

class Logger:
    @staticmethod
    def log(message, ts: bool = False):
        if ts is True:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}')
        else:
            print(f'{message}')

    @staticmethod
    def notify(message: str, ts: bool = False):
        Logger._log_level(message, LogLevel.Notification, ts=ts)

    @staticmethod
    def error(message: str, ts: bool = False):
        Logger._log_level(message, LogLevel.Error, ts=ts)

    @staticmethod
    def info(message: str, ts: bool = False):
        Logger._log_level(message, LogLevel.Info, ts=ts)

    @staticmethod
    def warning(message: str, ts: bool = False):
        Logger._log_level(message, LogLevel.Warning, ts=ts)

    @staticmethod
    def _log_level(message: str, level : LogLevel, ts: bool = False):
        prefix = level.value[1]

        if ts is True:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {prefix} {message}')
        else:
            print(f'{prefix} {message}')
