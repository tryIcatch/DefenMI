# DefenMI: Security Outsourcing Defense Against Membership Inference Attacks

## Overview
This repository contains the implementation of the **DefenMI** framework, proposed in the paper "DefenMI: Security Outsourcing Defense Against Membership Inference Attacks." The framework is designed to defend against membership inference attacks using a novel approach.

## Repository Structure
- **`train_user_classfication_model.py`**: Script to train the user classification model.
- **`train_defense_model_defensemodel.py`**: Script to train the defense model.
- **`train_attack_shadow_model.py`**: Script to train the shadow model for simulating attacks.
- **`evaluate_nn_attack.py`**: Script to evaluate the effectiveness of the defense against membership inference attacks.
- **`defense_framework.py`**: Core script implementing the defense tasks.
- **`config.ini`**: Configuration file to manage settings and parameters.
- **`run_location_defense.py`**: Script to run the full attack-defense pipeline.

## Installation
This project uses Python 3.9 and TensorFlow 2.6.0. It is recommended to use a Conda environment for managing dependencies.

### Steps to set up the environment:
1. Clone this repository:
   ```bash
   git clone https://github.com/tryIcatch/DefenMI.git
   cd DefenMI
   ```
2. Create a virtual environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the virtual environment:
   ```bash
   conda activate <DefenMI>
   ```

## Usage

### Training
1. Train the user classification model:
   ```bash
   python train_user_classfication_model.py
   ```
2. Train the defense model:
   ```bash
   python train_defense_model_defensemodel.py
   ```
3. Train the shadow model:
   ```bash
   python train_attack_shadow_model.py
   ```

### Evaluation
Evaluate the defense effectiveness:
```bash
python evaluate_nn_attack.py
```

### Full Pipeline
Run the complete attack-defense pipeline:
```bash
python run_location_defense.py
```

## Configuration
The `config.ini` file contains all configuration parameters required for the scripts. Update the file as needed before running any script.

## Requirements
- Python 3.9
- TensorFlow 2.6.0
- Dependencies specified in `environment.yml`

## Citation
If you use this code or the methodology in your research, please cite the original paper:

```
@article{DefenMI,
  title={DefenMI: Security Outsourcing Defense Against Membership Inference Attacks},
  author={hua shen,haocheng jiang},
  journal={Information Processing & Management},
  year={2025}
}
```

## Acknowledgments
Special thanks to the authors of "DefenMI" for their contribution to advancing security against membership inference attacks.

