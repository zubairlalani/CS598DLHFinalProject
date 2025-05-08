# CFVAE Reproduction — CS598 DLH Final Project (Spring 2025)

This repository contains our implementation and reproduction attempt of the paper:  
**_Explaining a Machine Learning Decision to Physicians via Counterfactuals_**  
📄 [Nagesh & Chen, PMLR 2023](https://proceedings.mlr.press/v209/nagesh23a/nagesh23a.pdf)  
🔗 [Original GitHub Repository](https://github.com/supriyanagesh94/CFVAE)

---

## 📚 Project Overview

As part of our final project for **CS598 Deep Learning for Healthcare**, we aimed to reproduce the results and core experiments of the CFVAE paper. The original work introduces a counterfactual explanation framework for clinical time-series data using a variational autoencoder architecture (CFVAE).

Our reproduction covers:

- Reimplementation and adaptation of CFVAE models for both **MIMIC-III** and **MNIST** datasets  
- Preprocessing pipeline reconstruction using **MIMIC-Extract** and public **MIMIC-III Demo data**  
- Training and evaluation of the **Multitask baseline model**  
- Generation of counterfactual examples using the CFVAE approach  
- Analysis of challenges in reproducibility (e.g., missing components, limited data access)

---

## 🚀 Quick Start (via Google Colab)

To run our experiments quickly and easily:

1. **Open** [`DLHFinalProject.ipynb`](DLHFinalProject.ipynb) in [Google Colab](https://colab.research.google.com/)
2. Run the **first cell** to clone the original CFVAE GitHub repository into the runtime.
3. All necessary demo data is included and will be accessible from the notebook.
4. Follow the section titles and comments throughout the notebook to explore:
   - MIMIC-III preprocessing
   - Multitask model training
   - CFVAE training and counterfactual generation
   - MNIST-based counterfactual experiments

---

## 📁 Repository Structure

```
CS598DLHFinalProject/
│
├── CFVAE/ # Original CFVAE GitHub repo (as subfolder)
├── extract/ # Output from running MIMIC-Extract
├── Mimic3DemoData.zip # Public demo version of MIMIC-III dataset
├── MNISTDataset.zip # Data for MNIST experiments
├── MNISTClassifier.py # MNIST classifier training and CFVAE usage
├── simple_impute.py # Basic imputation used in preprocessing
├── DLHFinalProject.ipynb # Main Colab notebook with all experiments
├── Resources.txt # List of useful resources and links
├── CS598_ProjectProposal.pdf # Project proposal (initial submission)
└── README.md # This file
```

---

## ⚠️ Notes on Reproducibility

- Due to restricted access to the full **MIMIC-III dataset**, we used the **MIMIC-III Demo** set to replicate the pipeline.
- Some components from the original codebase (e.g., time-series models, preprocessing scripts) were missing, requiring us to reconstruct them based on the paper's descriptions.
- As a result, full quantitative replication was not always possible, though core architecture and methodology were reproduced successfully.

---

## 🧠 Key Learnings

- Importance of complete and documented codebases in reproducible ML research
- Challenges of working with healthcare data (access, preprocessing, model alignment)
- Hands-on experience with counterfactual explanations and deep generative models

---

## 📬 Contact

For any questions or collaboration inquiries, feel free to reach out via GitHub or email.
