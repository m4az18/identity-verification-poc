import json
import logging
from datetime import UTC, datetime

_STANDARD_KEYS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    """Emit compact JSON logs suitable for Docker/centralized log collection."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_KEYS and key not in {"message", "asctime"}:
                is_scalar = isinstance(value, (str, int, float, bool, type(None)))
                payload[key] = value if is_scalar else str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure process logging once without including extracted identity content."""

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
