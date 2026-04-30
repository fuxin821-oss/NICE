# NICE: Neural Intercede Concept Erasure for Secure Diffusion Models

## 📖 Introduction
The rise of text-to-image diffusion models necessitates robust safety mechanisms capable of excising copyrighted, offensive, or unsafe concepts without compromising generative utility. However, current erasure techniques operate under critical limitations across semantic, structural, and generative safety dimensions. Existing approaches often neglect identity-style entanglement, causing global feature collapse, while weak constraints lead to unintended semantic drift. Furthermore, lacking hierarchical oversight degrades fine-grained background structures and leaves models susceptible to adversarial concept restoration. 

To address these challenges, we present **NICE**, a Neural Intercede Concept Erasure framework for secure diffusion models, advocating a multi-granular protection strategy from embedding space to feature maps. Experiments demonstrate that NICE achieves a superior trade-off between precise concept removal and non-target knowledge preservation.

<!-- <img width="4822" height="3529" alt="1teaser_01(1)" src="https://github.com/user-attachments/assets/6544a0df-7306-4c37-8e92-d4e110852db1" /> -->


## ✨ Key Features
Our framework achieves thorough erasure while maintaining generative integrity through four core modules:
*   **Non-Erasable Features Protector (NEFP)**: Reconstructs lost style attributes via orthogonal subspace decomposition to prevent global style shifts.
*   **Augmented Invariant Constraints (AIC)**: Enforces semantic stability through spectral analysis of concept clusters to avoid unintended semantic drift.
*   **Multi-level Semantic Protector (MSP)**: Aligns internal U-Net features across varying abstraction levels to safeguard local details and structural fidelity.
*   **Anti-Editing Mechanism (AEM)**: Prevents adversarial recovery attacks by injecting negative guidance into the unconditional prediction path.

## 📂 Repository Structure
The repository contains the following files and directories:
*   `LICENSE`: The license file for the project.
*   `README.md`: This documentation file.
*   `environment.yaml`: Conda environment configuration file for dependencies.
*   `inference_demo.py`: Demo script for running inferences with the edited model.
*   `src/`: Directory containing source code modules.
    *   `src/utils.py`: Utility functions and helper scripts.

## 🚀 Installation
You can set up the required environment using the provided YAML file:
```bash
conda env create -f environment.yaml
conda activate <your_env_name>
```

## 💻 Usage Examples

Below are examples of how to train and sample using our framework based on the provided scripts.

### 1. Concept Erasure (e.g., Instance Erasure)
To erase a specific target concept such as "Snoopy":
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --target_concepts "Snoopy" \
    --anchor_concepts "" \
    --retain_path "data/instance.csv" \
    --heads "concept"
```
To generate samples and evaluate the erased instance:
```bash
CUDA_VISIBLE_DEVICES=0 python inference_demo.py \
    --erase_type 'instance' \
    --target_concept 'Snoopy' \
    --contents 'Snoopy' \
    --mode 'original, edit' \
    --edit_ckpt 'logs/checkpoints_measure/NICE/Snoopy.pt' \
    --num_samples 30 \
    --batch_size 6 \
    --save_root 'results/XXX'
```

### 2. Style Erasure
To erase a specific style such as "Van Gogh":
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --target_concepts "Van Gogh" \
    --anchor_concepts "art" \
    --retain_path "data/style.csv" \
    --heads "concept"
```
To sample from the style erased model:
```bash
CUDA_VISIBLE_DEVICES=0 python inference_demo.py \
    --erase_type 'style' \
    --target_concept 'Van Gogh' \
    --contents 'Van Gogh' \
    --mode 'original, edit' \
    --edit_ckpt 'logs/checkpoints_measure/NICE/Van Gogh.pt' \
    --num_samples 30 \
    --batch_size 6 \
    --save_root 'results/XXX'
```

*Note: You can also utilize the included `inference_demo.py` to test generation capabilities natively.*
