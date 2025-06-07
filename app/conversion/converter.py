class Converter(object):
    def convert(self, entity, time_set: int = None, new_freq: str = None):
        raise NotImplementedError("Subclasses must implement 'convert'.")
