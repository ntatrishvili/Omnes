from app.conversion.converter import Converter


class PulpConverter(Converter):
    def convert(self, entity, time_set: int = None, new_freq: str = None):
        """
        Convert an Entity and its sub-entities into a flat dictionary of pulp variables
        suitable for optimization.

        This method recursively traverses the entity hierarchy, resamples all time series
        data to the specified frequency, and converts each TimeseriesObject into pulp-compatible
        variables using its `to_pulp` method.

        Parameters:
        ----------
        entity : Entity
            The root entity to convert (may have sub-entities).
        time_set : int, optional
            The number of time steps to represent in the pulp variables.
        new_freq : str, optional
            The target frequency to resample time series data to (e.g., '15min', '1H').

        Returns:
        -------
        dict
            A flat dictionary containing all pulp variables from the entity and its descendants.
        """
        variables = {
            key: ts.to_pulp(name=key, freq=new_freq, time_set=time_set)
            for key, ts in entity.quantities.items()
        }
        for sub_entity in entity.sub_entities:
            variables.update(self.convert(sub_entity, time_set, new_freq))
        return variables
