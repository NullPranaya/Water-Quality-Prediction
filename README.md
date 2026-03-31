# Water Quality Prediction

A project to predict water quality based on naturalistic and human-caused variables using EPA water quality data, USDA agricultural data, and climate data.

---

## What this project is

A collection of Jupyter notebooks that clean and prepare data from three sources:
- EPA water quality station measurements
- USDA NASS agricultural data
- Climate data

The notebooks are in `src/cleaning/tabular/`.

---

## Requirements

- Python 3.x
- Jupyter (to run the notebooks)

---

## Setup

**1. Install dependencies**

Open a terminal in the project folder and run:

```
pip install -r requirements.txt
```

**2. Install Jupyter if you don't have it**

```
pip install jupyter
```

---

## How to open and run the notebooks

**Option A: VS Code (recommended if you already have it)**

1. Open VS Code
2. Open the project folder: File > Open Folder > select `Water-Quality-Prediction`
3. Install the "Jupyter" extension if prompted
4. Navigate to any `.ipynb` file in `src/cleaning/tabular/`
5. Click "Run All" at the top of the notebook

**Option B: Jupyter in the browser**

1. Open a terminal in the project folder
2. Run:
   ```
   jupyter notebook
   ```
3. A browser window will open
4. Navigate to `src/cleaning/tabular/` and click any `.ipynb` file to open it
5. Click Kernel > Restart and Run All to run the whole notebook

---

## Notebook order (what to run and when)

The notebooks are independent cleaners for each data source. You can run them in any order, but a logical order is:

1. `src/cleaning/tabular/water-quality/epa-stations-clean.ipynb` - cleans EPA station location data
2. `src/cleaning/tabular/water-quality/epa-wq-clean.ipynb` - cleans EPA water quality measurements
3. `src/cleaning/tabular/water-quality/epa-merge-station-and-wq.ipynb` - merges the two EPA datasets
4. `src/cleaning/tabular/climate/climate-clean.ipynb` - cleans climate data
5. `src/cleaning/tabular/agricultural/usdaNass-agriculture-clean.ipynb` - cleans agricultural data

---
