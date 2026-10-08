# NASA C-MAPSS Dataset Instructions

### Primary Dataset Mirror (Kaggle)
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

### NASA Prognostics Center of Excellence (PCoE)
https://data.nasa.gov/

---

## Dataset Overview
The NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS) contains 4 operational sub-datasets:
- **FD001**: 100 Train / 100 Test engines | 1 Operating Condition (Sea Level) | 1 Fault Mode (HPC Degradation)
- **FD002**: 260 Train / 259 Test engines | 6 Operating Conditions (Full Flight Envelope) | 1 Fault Mode (HPC Degradation)
- **FD003**: 100 Train / 100 Test engines | 1 Operating Condition (Sea Level) | 2 Fault Modes (HPC + Fan Degradation)
- **FD004**: 249 Train / 248 Test engines | 6 Operating Conditions (Full Flight Envelope) | 2 Fault Modes (HPC + Fan Degradation)

## Data Location in Repository
All 12 canonical raw `.txt` files are located in:
```
project/data/raw/
```
Expected files:
- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

> **Note**: All 12 files are pre-packaged and directly included in this repository. When you clone or download this project, all datasets are ready to run immediately with zero external downloading required.