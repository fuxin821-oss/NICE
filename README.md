# NICE
Preview repository for NICE, a concept erasure framework for safer text-to-image diffusion models.
<img width="4822" height="3529" alt="1teaser_01(1)" src="https://github.com/user-attachments/assets/6544a0df-7306-4c37-8e92-d4e110852db1" />


## Introduction

Text-to-image diffusion models have achieved remarkable progress in high-quality visual generation. However, their strong generative capability also raises safety concerns, such as the generation or misuse of copyrighted, sensitive, or unsafe concepts. Concept erasure has therefore become an important direction for improving the safety and controllability of diffusion models.

Despite recent advances, existing concept erasure methods still struggle to balance effective concept removal and the preservation of non-target knowledge. Over-aggressive erasure may lead to semantic drift, structural degradation, object disappearance, or reduced generation quality, while insufficient erasure may allow the target concept to be recovered through adversarial prompts or malicious image editing.

To address these challenges, we propose **NICE**, a secure concept erasure framework for diffusion models. NICE aims to suppress target concepts while preserving non-target semantics, structural details, and generative utility. It further improves the robustness of erased models against malicious editing and unintended concept restoration.

## Method Overview

NICE adopts a multi-level protection strategy to progressively improve the stability and safety of concept erasure from representation space to generative features. The overall idea includes:

- Protecting non-target features to reduce unintended damage to information such as style during concept erasure;
- Constraining semantic stability during generation to mitigate non-target concept drift and object disappearance;
- Preserving multi-level structural information to alleviate background and local region degradation;
- Enhancing the model’s resistance to malicious editing and concept restoration attacks.


