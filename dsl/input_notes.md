## 0. Unit
- `id` (str): Unique identifier of the component

## 1. Component (Unit)
General class for devices (producers, consumers, prosumers)
- `meter` (str): ID of the meter the component is connected to
- `bus` (str): ID of the bus the component is connected to
- `phase` (int, 0-2): The id of the phase the component connects to 

### 1.1 Generator (Component)
Produces electricity.
- `type` (enum): PV, Wind, FuelCell
- `peak_power` (float, kWp): Maximum output of the generator
- `cosphi` (float, 0-1): Power factor
- `efficiency`:
  - `base` (float, 0-1)
  - `curve_file` (optional, str): Efficiency as function of time
- `timeseries_output` (str): File of expected output (if nondispatchable, in kWh)
- `dispatchable` (bool)

### 1.2 Energy Storage System (Component)
Stores energy.
- `capacity` (float, kWh)
- `initial_soc` (float, kWh)
- `soc_min` (float, kWh)
- `soc_max` (float, kWh)
- `charge_efficiency`:
  - `base` (float, 0-1)
  - `degradation_curve` (optional, str)
- `discharge_efficiency`: same structure as above
- `storage_efficiency`: same structure as above
- `max_charge_rate` (float, kW)
- `max_discharge_rate` (float, kW)
- `charging_device` (str): ID of the Load of the charging device
- `lifetime_cycles` (int)
- `calendar_lifetime` (years)

Dynamic properties: state_of_charge, charge_power, discharge_power

#### 1.2.1 Battery (Energy Storage System)
Specialization for electrochemical storage.

#### 1.2.2 Thermal Energy Storage (Energy Storage System)
- `material` (enum): Water, Air
- `volume` (float, m3)
- `set_temperature` (float or str): °C or file
- `input_temperature`, `output_temperature` (same)
- `heat_loss_rate` (optional, float, W or %/h)

##### 1.2.2.1 Hot Water Storage (Thermal Storage)
- `max_flow_rate` (float, m3/s)

### 1.3 Load (Component)
Energy consumer.
- `power_type` (enum): I, Z, P
- `service_type` (enum): heat, cooling, electric, EV, appliance
- `timeseries_input` (str): kWh consumption
- `cosphi` (float, 0-1)
- `demand_response` (bool)
- `priority` (optional, int)

### 1.4 Grid (Component)
Connection to national grid.
- `bus` (str): The bus the grid is connected to

Dynamic properties: energy_in, energy_out

---

## 2. Electric Component (Unit)

### 2.1 Bus (Electric Component)
- `type` (enum): PQ, Slack, I, Z
- `household` (optional, str)
- `voltage` (float, V)
- `phase_count` (int): 1 or 3
- `allowed_voltage_range` (tuple, optional)

Dynamic properties: voltage, P, Q

### 2.2 Line (Electric Component)
- `from_bus` / `to_bus` (str)
- `resistance` (float, Ohm)
- `reactance` (float, Ohm)
- `line_length` (float, km)
- `max_current` (float, A)
- `status` (bool): Online/offline
- `line_type` (enum): overhead, cable

---

## 3. Computation Model
- `time_steps` (int)
- `time_resolution_value` (float)
- `time_resolution_unit` (enum): s, min, h, day, ...

### 3.1 Optimization Model (Computation Model)
- `model_type` (enum): LP, MILP, MINLP, ...
- `objective_function` (enum): Cost min, CO2 min, Self-sufficiency max, Grid interaction min
- `solver` (str): CPLEX, Gurobi, etc.
- Other TBD

### 3.2 Simulation Model (Computation Model)
- `type` (enum): State estimation, Power flow
- `api` (enum): pandapower, TBD

