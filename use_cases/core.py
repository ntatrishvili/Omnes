import xarray as xr
from abc import ABC, abstractmethod

class Element:
    def __init__(self, name: str = None, quantities: dict = None,
                 relations: dict = None, attrs: dict = None):
        self._name = name
        self._quantities = {}
        self._relations = {}
        self._attrs = attrs or {}

        # Validate and add quantities if provided.
        if quantities:
            if not isinstance(quantities, dict):
                raise TypeError(
                    "quantities must be a dict of {name: DataArray}")
            # Using ** so you can pass keyword arguments.
            self.add_quantities(**quantities)

        # Validate and add relations if provided.
        if relations:
            if not isinstance(relations, dict):
                raise TypeError(
                    "relations must be a dict of {name: relation_string}")
            self.add_relations(relations)

    # ------------------------
    # Property Accessors
    # ------------------------

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value) if value is not None else None

    @property
    def attrs(self):
        # No setter: treat as read-only (modify in place if needed)
        return self._attrs

    @property
    def quantities(self):
        # Exposes a read-only view of quantities.
        return self._quantities

    @property
    def relations(self):
        # Exposes a read-only view of relations.
        return self._relations

    # ------------------------
    # Methods for Adding Data
    # ------------------------

    def add_quantities(self, **quantities):
        """
        Add one or more quantities by keyword.
        Usage:
            element.add_quantities(pv=da1, load=da2)
        """
        for name, data in quantities.items():
            self._check_quantity(name, data)
            self._quantities[name] = data

    def add_relations(self, relations: dict):
        """
        Add multiple relations from a dictionary.
        The relations dictionary should be of the form:
          { relation_name: relation_string, ... }
        For example:
          {'output_power': "p_pv = size * yield"}
        """
        for name, relation in relations.items():
            self._check_relation(name, relation)
            self._relations[name] = relation

    # ------------------------
    # Getters for Individual Items
    # ------------------------

    def get_quantity(self, name):
        return self._quantities.get(name)

    def get_relation(self, name):
        return self._relations.get(name)

    # ------------------------
    # Internal Checks
    # ------------------------

    def _check_quantity(self, name, data):
        if name in self._quantities:
            raise ValueError(f"Quantity '{name}' already exists.")
        if not isinstance(data, xr.DataArray):
            raise TypeError(f"Quantity '{name}' must be an xarray.DataArray.")

    def _check_relation(self, name, relation):
        if name in self._relations:
            raise ValueError(f"Relation '{name}' already exists.")
        # You can insert further validation logic here if needed.

    # ------------------------
    # Representation
    # ------------------------

    def __repr__(self):
        return (
            f"<Element '{self._name}': {len(self._quantities)} quantities, "
            f"{len(self._relations)} relations, {len(self._attrs)} attrs>")

class TimeContext:
    def __init__(self, time_index, resolution, horizon):
        self.time_index = time_index      # list or array of time stamps
        self.resolution = resolution      # e.g., 'H' for hourly, 'Y' for yearly, etc.
        self.horizon = horizon            # total time period

    def interpolate_quantity(self, quantity):
        # Pseudocode: adjust quantity to match the time_index if necessary
        # This might use xarray's interpolation methods, reindexing, etc.
        return quantity


# class ProblemBuilderBase(ABC):
#     def __init__(self, elements, time):
#         self.elements = elements  # list of Elements
#         self.time = time          # time object: horizon, resolution, etc.
#
#     @abstractmethod
#     def build(self):
#         pass

# class ProblemDefinition:
#     def __init__(self, variables, parameters, constraints, time):
#         self.variables = variables
#         self.parameters = parameters
#         self.constraints = constraints
#         self.time = time

# class MILPProblemBuilder(ProblemBuilderBase):
#     def __init__(self, elements, time, variable_map=None):
#         super().__init__(elements, time)
#         self.variable_map = variable_map or {}  # Optional mapping: {'element_name': ['q1', 'q2']}
#
#     def build(self):
#         variables = {}
#         parameters = {}
#         constraints = []
#
#         for el in self.elements:
#             for q_name, q_data in el.get_quantities().items():
#                 if self._is_variable(el.name, q_name):
#                     variables[(el.name, q_name)] = q_data
#                 else:
#                     parameters[(el.name, q_name)] = q_data
#
#             for rel_name, rel_expr in el._relations.items():
#                 constraints.append((el.name, rel_name, rel_expr))
#
#         return ProblemDefinition(variables, parameters, constraints, self.time)
#
#     def _is_variable(self, el_name, q_name):
#         return self.variable_map.get(el_name, []) and q_name in self.variable_map[el_name]

# class SimulationBuilder(ProblemBuilderBase):
#     def __init__(self, elements, time, dependency_map=None, controller=None):
#         super().__init__(elements, time)
#         self.dependency_map = dependency_map or {}  # Which quantities are dependent
#         self.controller = controller                # Callable or class controlling independent vars
#
#     def simulate(self):
#         results = {}
#         for t in self.time.get_steps():
#             input_values = self.controller.get_inputs(t)
#             current_state = self._evaluate_step(t, input_values)
#             results[t] = current_state
#         return results
#
#     def _evaluate_step(self, t, inputs):
#         # Step-by-step calculation using relations
#         # Evaluate all dependent variables
#         # Possibly use xarray and symbolic parsing
#         return NotImplemented