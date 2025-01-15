# Efficient-SVTR: Lightweight Scene Text Recognition with Enhanced Robustness

Welcome to the official repository for **Efficient-SVTR**, a lightweight and efficient framework designed for **Scene Text Recognition (STR)**. This repository hosts the implementation of **SVTR-Compact**, a novel approach that enhances the baseline SVTR model through two key innovations: the **Jumble Module** and the **Self-Distillation Module**. Our work achieves **state-of-the-art performance** on benchmark datasets while maintaining computational efficiency, making it ideal for deployment in resource-constrained environments.

This work is currently under submission to **Springer Nature's *The Visual Computer***, a prestigious journal for computer vision and graphics research. We are excited to share our **code**, **pre-trained models**, and **evaluation protocols** to support reproducibility and foster further advancements in the field of STR.

---

## Key Features

- **Jumble Module**:  
  Enhances robustness against spatial distortions by introducing random spatial perturbations before the patch embedding stage. This module significantly improves the model's ability to handle distorted, rotated, or perspective-transformed text.

- **Self-Distillation Module**:  
  Integrated into the SVTR framework, this module refines feature representations and compresses the model size without sacrificing accuracy. It enables efficient deployment on devices with limited computational resources.

- **State-of-the-Art Performance**:  
  Comprehensive experiments on benchmark datasets demonstrate that SVTR-Compact outperforms existing methods, achieving new benchmarks in both accuracy and efficiency.

- **Lightweight Design**:  
  SVTR-Compact strikes an optimal balance between computational efficiency and recognition accuracy, making it suitable for real-world applications in resource-constrained environments.

---

## Repository Contents

- **Code**: Full implementation of SVTR-Compact, including the Jumble Module and Self-Distillation Module.
- **Pre-trained Models**: Ready-to-use models for quick deployment and evaluation.
- **Evaluation Protocols**: Scripts and guidelines for reproducing our experimental results.
- **Datasets**: Instructions for downloading and preparing benchmark datasets used in our experiments.

---

## Installation

To get started, clone this repository and install the required dependencies:

```shell
git clone https://github.com/lingyurou/Efficient-SVTR.git
cd Efficient-SVTR
pip install -r requirements.txt
```
Usage
Training
To train the SVTR-Compact model on your dataset, run:

```shell
python train.py --config path/to/config.yaml
```
Evaluation
To evaluate a pre-trained model on a benchmark dataset, use:

```shell
python evaluate.py --model path/to/model.pth --dataset path/to/dataset
```
Inference
For text recognition on custom images, run:

```shell
python infer.py --model path/to/model.pth --image path/to/image.jpg
```
Results
| Method                        | ACC   | NED   |
|-------------------------------|-------|-------|
| CRNN                          | 53.4  | -     |
| ASTER                         | 54.5  | -     |
| MORAN                         | 51.8  | -     |
| SAR                           | 62.5  | -     |
| SRN                           | 60.1  | -     |
| SVTR (Baseline)               | 68.5  | 84.7  |
| **SVTR-Compact (Ours)**       | **70.0** | **84.9** |

**Table 1:** Results on Benchmarking Chinese Text Recognition Scene Dataset.

Citation
If you find our work useful, please consider citing our paper (to be updated upon acceptance):

```bibtex
@article{lingyurou2023efficientsvtr,
  title={Efficient-SVTR: Lightweight Scene Text Recognition with Enhanced Robustness},
  author={Ling, Yurou and Collaborators},
  journal={The Visual Computer},
  publisher={Springer Nature},
  year={2023},
  note={Under Submission}
}
```
Contributing
We welcome contributions from the community! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request. For major changes, please discuss them with us first by creating an issue.
