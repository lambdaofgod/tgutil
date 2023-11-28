from enum import Enum


class StrEnum(str, Enum):
    @classmethod
    def contains(cls, elem):
        return elem in [m.value for m in cls]
