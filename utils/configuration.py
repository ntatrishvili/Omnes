import logging
from configparser import RawConfigParser, ExtendedInterpolation
from os import getcwd
from os.path import join

import pandas as pd
from app.infra.singleton import Singleton

logger = logging.getLogger(__name__)


class Config(Singleton):
    def __init__(self, config_filename=join(getcwd(), "..", "config", "config.ini")):
        super().__new__(self)
        self.__config = RawConfigParser(
            allow_no_value=True, interpolation=ExtendedInterpolation()
        )
        self.__config.read_file(open(config_filename))
        self._registered_entries = {"time": {"frequency": self._process_frequency}}

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def _process_frequency(self):
        freq_str = self.__config.get("time", "frequency")
        try:
            return pd.Timedelta(freq_str)
        except ValueError:
            return freq_str

    def get(self, section, key, fallback=None):
        if (
            section not in self._registered_entries
            or key not in self._registered_entries[section]
        ):
            return self._get(section, key, fallback)
        return self._registered_entries[section][key]()

    def _get(self, section, key, fallback=None):
        try:
            value = self.__config.get(section, key, fallback=fallback)
        except Exception as e:
            raise KeyError(
                f"Section '{section}', key '{key}' problem in configuration: '{e}'"
            )
        if value is None:
            raise KeyError(
                f"Section '{section}', key '{key}' not found in configuration"
            )

        if "," not in value:
            return value

        values = list(filter(len, value.strip("][").split(",")))
        all_integers = all(element.isdigit() for element in values)
        if all_integers:
            return [int(x) for x in values]

        return [v.replace(" ", "") for v in values]

    def set(self, section, key, value):
        if type(value) != str:
            value = str(value)
            logger.warning(
                f"Configuration is set with a non-string-type value: {value}"
            )
        self.__config.set(section, key, value)

    def setint(self, section, key, value):
        self.__config.set(section, key, f"{value}")

    def setarray(self, section, key, value):
        if len(value) == 1:
            set_str = f"{value[0]},"
        else:
            set_str = f",".join(f"{v}" for v in value)
        self.__config.set(section, key, set_str)

    def setboolean(self, section, key, value):
        boolean_str = "True" if value else "False"
        self.__config.set(section, key, boolean_str)

    def getboolean(self, section, key, fallback=None):
        return self.__config.getboolean(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        try:
            return self.__config.getint(section, key, fallback=fallback)
        except:
            return self.__config.get(section, key, fallback=fallback)

    def getstr(self, section, key, fallback=None):
        return self.__config.get(section, key, fallback=fallback)

    def getarray(self, section, key, dtype=str, fallback=None):
        val = self._get(section, key, fallback=fallback)
        try:
            return [dtype(v) for v in val]
        except TypeError:
            return [
                dtype(val),
            ]

    def getfloat(self, section, key, fallback=None):
        return self.__config.getfloat(section, key, fallback=fallback)

    def has_option(self, section, option):
        return self.__config.has_option(section, option)

    def set_and_check(self, section, key, value, setter=None, check=True):
        if check and setter is None:
            value_cf = self.getfloat(section, key)
            if value != value_cf:
                logger.warning(
                    f"The value of [section={section}, key={key}] set dynamically (value={value}) does not equal "
                    f"the original value from the configuration file (value={value_cf})"
                )
        if setter is None:
            if type(value) == int:
                self.setint(section, key, value)
            else:
                self.set(section, key, value)
        else:
            # Get the setter function passed using the current Configuration object
            getattr(self, setter.__name__)(section, key, value)
