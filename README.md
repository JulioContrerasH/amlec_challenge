<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Language](https://img.shields.io/badge/Language-Python_|_English-blue)
![Task](https://img.shields.io/badge/Task-Emulation-green)
![Domain](https://img.shields.io/badge/Domain-Atmospheric_Radiative_Transfer-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

<br/>
<br/>


<div align="center">
  <img src="https://elias-ai.eu/wp-content/uploads/2023/09/elias_logo_big-1.png" alt="elias_logo" style="width:30%; margin-right: 50px;">
  <img src="https://elias-ai.eu/wp-content/uploads/2024/01/EN_FundedbytheEU_RGB_WHITE-Outline-1.png" alt="eu_logo" margin-left: 50px; style="width:50%;">
</div>

</div>


# **Atmospheric Radiative Transfer Emulation Challenge (AMLEC)**

> **Reference paper:** *Evaluating Machine Learning Emulators for Atmospheric Radiative Transfer: The AMLEC Challenge* 
> **Authors:** Jorge Vicent, Jasdeep Singh, Axel Rochel, Julio Conteras, Panagiotis Liatsis, Hasan Al Marzouqi, and Gustau Camps-Valls.

## **Overview & abstract**

The **Atmospheric Machine Learning Experiment Competition (AMLEC)**, organized within the EU ELIAS project, provided a benchmark for evaluating machine learning approaches to emulating atmospheric radiative transfer models. 

Participants were tasked with predicting spectral data across two scenarios involving different input variables and spectral configuration: 
1. **Scenario A:** Atmospheric correction of hyperspectral satellite data.
2. **Scenario B:** CO2 concentration retrieval.

Several training datasets, covering realistic input ranges with 500 to 10,000 samples, were used. Testing included interpolation and extrapolation to out-of-range conditions. Eight models were submitted, spanning neural networks and Gaussian processes with various configurations. Results showed that Gaussian process approaches achieved the lowest errors, indicating their suitability while highlighting the challenge of training complex neural network approaches with scarce data. 

This repository serves as the permanent archive for the challenge data, the submission evaluation code, and the final benchmark results.

---

## **Benchmark results**

| **Model** | **MRE A1 (%)** | **MRE A2 (%)** | **MRE B1 (%)** | **MRE B2 (%)** | **Score** | **Runtime** | **Rank** |
|-----------|---------------|---------------|---------------|---------------|----------|----------|--------|
| Jasdeep_Emulator_3 | 0.090 | 3.117 | 0.566 | 6.108 | 1.525 | 89.359 | 1° |
| Hugo2 | 0.144 | 2.868 | 0.610 | 5.033 | 2.300 | 5.382 | 2° |
| rpnn1 | 0.133 | 5.883 | 0.583 | 5.561 | 2.525 | 19.082 | 3° |
| rpgprv2 | 0.176 | 3.835 | 0.640 | 7.050 | 4.000 | 35.650 | 4° |
| Jasdeep_Emulator_2 | 0.886 | 3.895 | 0.768 | 6.176 | 5.625 | 2.078 | 5° |
| Krtek | 0.545 | 7.693 | 0.823 | 7.877 | 6.500 | 0.764 | 6° |
| rpcvae | 0.185 | 11.996 | 0.918 | 15.313 | 6.700 | 0.546 | 7° |
| Jobaman1 | 0.296 | 10.093 |  | 23.258 | 7.675 | 6.150 | 8° |
| baseline | 0.998 | 12.604 | 1.084 | 7.072 | 8.150 | 0.241 | 9° |

---

## **Introduction**

Atmospheric Radiative Transfer Models (RTM) are crucial in Earth and climate sciences with applications such as synthetic scene generation, satellite data processing, or numerical weather forecasting. However, their increasing complexity results in a computational burden that limits direct use in operational settings. 

RTM emulation is challenging due to the high-dimensional nature of both input (~10 dimensions) and output (several thousand) spaces, and the complex interactions of electromagnetic radiation with the atmosphere. This challenge contributes to reducing computational burdens in climate and atmospheric research, enabling faster satellite data processing and improved accuracy in atmospheric correction.

## **Challenge tasks and data**

### **Proposed experiments**

1. **Atmospheric correction** (`A`): Focuses on reproducing key atmospheric transfer functions (path radiance, direct/diffuse solar irradiance, transmittance) for hyperspectral data (400-2500 nm).
2. **CO<sub>2</sub> column retrieval** (`B`): Focuses on predicting top-of-atmosphere radiance, particularly within the spectral range sensitive to CO<sub>2</sub> absorption (2000-2100 nm).

Each scenario-track combination is identified by `Sn`, where `S`={`A`,`B`} and `n`={1,2} (1=Interpolation, 2=Extrapolation).

### **Data format**

Training data is stored in **HDF5** format with `LUTdata` (outputs) and `LUTHeader` (inputs). Testing input datasets are stored in `.csv` format.

Example loading in Python:
```python
import h5py
import pandas as pd

# Load Training Data
with h5py.File('train2000.h5', 'r') as h5:
    Ytrain = h5['LUTdata'][:]
    Xtrain = h5['LUTHeader'][:]
    wvl = h5['wvl'][:]

# Load Testing Inputs
Xtest = pd.read_csv('refInterp.csv').to_numpy()
```


Data is available in the [repository files](https://huggingface.co/datasets/isp-uv-es/rtm_emulation).

## **Evaluation methodology**

### **Prediction accuracy**

* **Scenario A:** Mean Relative Error (MRE) of retrieved surface reflectance.
* **Scenario B:** MRE of predicted TOA radiance.
* **$MRE_\lambda$** excludes deep water vapor absorption bands.

### **Final Score**

The final ranking is a weighted average of the ranks in the four sub-tracks:


$$Score = 0.325 \cdot Rank_{A1} + 0.175 \cdot Rank_{A2} + 0.325 \cdot Rank_{B1} + 0.175 \cdot Rank_{B2}$$

-----

## **Reproducibility**

This repository contains the source code used to evaluate the challenge submissions.

### **Running the benchmark**

1.  Clone this repository.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Ensure you have the reference data available (see `config.py`).
4.  Run the evaluation script:
```bash
python main.py
```
*Note: This requires appropriate Hugging Face credentials if you intend to sync results with the hub.*

-----

## **Citation**

If you utilize this code, repository, or the provided datasets in your research, please cite the following publication:

> Vicent, J., et al. "Evaluating Machine Learning Emulators for Atmospheric Radiative Transfer: The AMLEC Challenge." [Journal/Proceedings Name], 2025. (Submitted)