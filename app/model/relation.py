class Relation:
    """
    Represents a semantic or mathematical relationship between the properties of entities, or between time steps of a single property.

    This class can be extended to express:
        - Mathematical constraints (e.g., entity_A.prop = 2 * entity_B.prop)
        - Conditional rules (e.g., "if storage.stored_energy_percent > .08 then...").
        - Temporal constraints (e.g., device must be active during daylight hours "device.on[10:16]=true").
    """

    pass
