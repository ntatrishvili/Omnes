[![CI Pipeline](https://github.com/ntatrishvili/Omnes/actions/workflows/ci.yml/badge.svg)](https://github.com/ntatrishvili/Omnes/actions/workflows/ci.yml)
[![CodeQL Analysis](https://github.com/ntatrishvili/Omnes/actions/workflows/codeql.yml/badge.svg)](https://github.com/ntatrishvili/Omnes/actions/workflows/codeql.yml)
[![Nightly](https://github.com/ntatrishvili/Omnes/actions/workflows/nightly.yml/badge.svg)](https://github.com/ntatrishvili/Omnes/actions/workflows/nightly.yml)
[![Docs](https://github.com/ntatrishvili/Omnes/actions/workflows/docs.yml/badge.svg)](https://github.com/ntatrishvili/Omnes/actions/workflows/docs.yml)

# Omnes 

**Omnes** is an energy system modelling platform that lets you transform structured datasets into solvable energy models using a simple configuration and processing pipeline. Built for flexibility and clarity, Omnes supports modular data uploads, transformation, and model execution — currently with support *PuLP* solver and *Pandapower* simulator.

---

## Features

- Upload data files to the `data/` folder to describe your energy system (e.g., technologies, resources, demands)
- Easily configure the time set, resolution, and model settings via `config.ini`
- Transform input data into a model-ready format
- Run your custom optimization script with full PuLP integration
- Run your custom simulation with Pandapower

---

## Repository Structure

```bash
Omnes/
├── .github/                # GitHub workflows and config (e.g. CI/CD)
├── app/                    # Core application logic
│   ├── conversion/         # Data transformation and conversion
│   ├── infra/              # Infrastructure utilities & helpers
│   ├── model/              # Data model components
│   ├── operation/          # Main modelling and optimization logic
│   └── __init__.py
├── data/                   # Input datasets and model configuration
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ntatrishvili/Omnes.git
cd Omnes
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: You must have a working [Gurobi license](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) and `gurobipy` installed.

### 3. Add your data

Place your system definition files (CSV, Excel, etc.) into the `data/` folder. These files represent your technologies, time series, costs, and other elements.

### 4. Configure model settings

Edit `config.ini` to set your **time resolution**, **frequency**, and other options:

```ini
[time]
time_set = 10000
frequency = H  # e.g., H for hourly, D for daily
```

### 5. Run the model

```bash
python main.py
```

This will transform the input data, build your energy system model, and run it using your configured script in `operation/`.

---

## Requirements

- Python 3.9+
- [Gurobi](https://www.gurobi.com/) (Academic or commercial license)
- `gurobipy`, `pandas`, `numpy`, `configparser`, etc. (see `requirements.txt`)

---

## Contributing

Contributions are welcome! Please open an issue or pull request to discuss improvements, bugs, or new features.

---

## License

MIT License — see `LICENSE` file.  
SPDX-License-Identifier: MIT

---

## How to cite
If you use this framework in academic work or reports, please cite it as:

>Tatrishvili N., Barancsuk L., Lorenti G., Optimization and Simulation of Multi-node Multi-Energy Systems (version 1.0), GitHub repository, 2025. Available at: https://github.com/ntatrishvili/Omnes

You can also use a BibTeX-style entry:
```LaTex
@misc{omnes,
  author       = {Tatrishvili N., Barancsuk L., Lorenti G.},
  title        = {Optimization and Simulation of Multi-node Multi-Energy Systems},
  year         = {2025},
  howpublished = {GitHub repository},
  note         = {Version 1.0},
  url          = {https://github.com/ntatrishvili/Omnes}
}
```

## Maintainer

Built and maintained by

-[Nia Tatrishvili](https://github.com/ntatrishvili)
-[Lilla Barancsuk](https://github.com/Lilol)
-[Gianmarco Lorenti](https://github.com/gianmarco-lorenti)

---

## 🎓 Acknowledgments

This project was developed as part of a **Bachelor's Thesis** at the  
**[Budapest University of Technology and Economics (BME)](https://www.bme.hu/)**,  
in collaboration with the **[Politecnico di Torino](https://www.polito.it/)** and the  
**[HUN-REN Centre for Energy Research](https://www.ek.hun-ren.hu/en/home/)**.
