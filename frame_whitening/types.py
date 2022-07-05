import enum


class MyEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        return item in [v.value for v in cls.__members__.values()]


class FuncType(enum.Enum, metaclass=MyEnumMeta):
    POWER = "POWER"
    EXPONENTIAL = "EXPONENTIAL"
    G_EXPONENTIAL = "G_EXPONENTIAL"
