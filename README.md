# MoFA-Lite

> **Lightweight Modality-Focused Attention for Multimodal Video Popularity Prediction**  
> *Developed for the RS_AI_2025 Final Examination*

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

## 📖 Introduction

Welcome to **MoFA-Lite**! This repository hosts the implementation of my solution and insights for the **Micro-Video Popularity Prediction (MVPP)** task.

### 📚 Task Background
The objective is to predict the popularity (views, likes, comments) of micro-videos over a specific period based on multimodal features.
*   **Dataset**: 8,000 training samples with ground truth labels (log-transformed).
*   **Input Features**:
    *   `image_features` (1024-dim): Visual features extracted from video covers using **CLIP-RN50**.
    *   `video_features` (768-dim): Motion/Content features extracted using [**VideoMAE**](http://arxiv.org/abs/2203.12602).
    *   `text_features` (1024-dim): Semantic features extracted from descriptions using [**BGE-M3**](http://arxiv.org/abs/2402.03216).

> **Acknowledgement**: All data sources were provided by the team of the course instructor, [Prof. Chen](https://zhenzhong-chen.github.io/), and are intended solely for the final course examination.

**MoFA-Lite** tackles this by employing a **Lightweight Modality-Focused Attention** mechanism that dynamically weighs the importance of different modalities before fusing them for prediction.

Unlike traditional heavy-weight transformers, MoFA-Lite is designed to be agile yet powerful, leveraging a **Single-Model Multi-Output** strategy combined with **10-Fold Cross-Validation Ensemble** to achieve state-of-the-art stability and accuracy.

## 📊 Performance & Leaderboard

Our solution achieved the **1st rank** on the course leaderboard, demonstrating superior generalization compared to other baselines.

| Model | View NMSE | Like NMSE | Comment NMSE | **Score** |
| :--- | :---: | :---: | :---: | :---: |
| **MoFA-Lite (Ours)** | **0.8071** | **0.9476** | **0.7831** | **0.8459** |
| Baseline 3 | 0.8284 | 0.9590 | 0.7935 | 0.8603 |
| Baseline 2 | 0.8591 | 0.9851 | 0.8332 | 0.8925 |
| Baseline 1 | 0.8708 | 1.0362 | 0.8873 | 0.9314 |
> Note: At the time of this repository's release, our solution stands as the SOTA on the leaderboard.

### 📉 Evaluation Metric
The performance is evaluated using **Normalized Mean Square Error (NMSE)**. The final score is the average NMSE across all three targets:

$$ Score = \frac{1}{3} (NMSE_{view} + NMSE_{like} + NMSE_{comment}) $$

*Lower Score indicates better performance.*

## 🧩 Related Work & Analysis

During the development of this project, several related works were referenced and reproduced:

-   [**MMRA**](https://dl.acm.org/doi/10.1145/3626772.3657929): Currently a mainstream SOTA solution. While the prediction results appeared normal, submission often resulted in format errors or required algorithmic adjustments, casting some doubt on its effectiveness in this specific context.
-   [**MMVED**](http://arxiv.org/abs/2003.12724): A common baseline solution released by Prof. Chen's group several years ago.
-   [**MVP**](http://arxiv.org/abs/2507.00950): The winning solution to the SMP Challenge 2025 Video Track. Its main insight involves averaging the outputs of a Neural Network and XGBoost. However, likely due to different evaluation metrics, our tests showed that introducing tree-based models like XGBoost did not yield significant improvements. We also believe this approach is more suited for time-series regression, which offers no distinct advantage in the current task.

**Major Challenges Encountered:**
The primary difficulties revolved around **feature utilization** and **overfitting**.

1.  **Data Preprocessing**: Initial strategies seemed flawed.
2.  **Small Dataset**: With only 8,000 samples, the model was highly prone to overfitting.
3.  **Feature Utilization**:
    *   Training separate deep learning models for each metric failed to capture the correlations between targets.
    *   Simple modal fusion felt too crude.
    *   Adding complex attention mechanisms (e.g., Cross-Modal Attention, Multi-Head Attention) surprisingly yielded poor results.

We considered using the two key insights from [**AdaptCLIP**](https://arxiv.org/abs/2505.09926):
> 1. Adaptive visual and textual representations should be learned alternately rather than jointly.
> 2. Comparative learning should incorporate contextual and aligned residual features rather than relying solely on residual features.

While this slightly improved performance, it resulted in an overly complex model.


## 💡 Key Features

To address these challenges (primarily overfitting), the following strategies were implemented:
*   **Lightweight Attention Mechanism**: Efficiently captures inter-modal dynamics without the computational overhead of massive self-attention blocks.
*   **Multi-Objective Prediction**: Simultaneously predicts `view_count`, `like_count`, and `comment_count` using a shared representation, exploiting the latent correlations between these metrics.
*   **Robust Preprocessing**: Implements **RankGauss** transformation for video features to handle long-tail distributions and outliers effectively.
*   **K-Fold Ensemble**: A rigorous cross-validation strategy that ensures the model generalizes well to unseen data, minimizing the risk of overfitting.
*   **Comprehensive Analysis**: Includes tools for detailed feature distribution analysis and visualization.

## ⚠️ Remaining Challenges

*   **Overfitting**: Despite employing techniques like K-Fold Ensemble, overfitting remains a significant issue. Training often triggers early stopping almost immediately (visible in the volatile loss curves), making it difficult to balance model size with effective feature utilization.
*   **Feature Distribution**: Our understanding of the feature distribution remains incomplete. While observing standard deviations suggested that VideoMAE features required extra processing, applying corrections based on the skewness of predictions (left-skewed views, right-skewed comments) did not yield satisfactory results.
*   **Outlier Handling**: Attempts to remove outliers using common sense (e.g., views < likes/comments) or statistical methods (IQR, feature norms) significantly degraded performance. We suspect that due to the small dataset size, removing "abnormal" data prevented the model from learning more generalized patterns.

## 🏗️ Model Architecture

The MoFA-Lite architecture follows a streamlined 4-stage pipeline:

1.  **Feature Encoding**: Independent MLPs project Image (1024d), Video (1024d), and Text (768d) features into a shared latent space ($D_{hidden}=512$).
2.  **Modality Attention**: A lightweight gating network calculates attention scores for each modality, allowing the model to "focus" on the most informative signals.
3.  **Fusion**: Weighted features are concatenated and passed through a fusion layer with LayerNorm and GELU activation.
4.  **Prediction Head**: A final MLP projects the fused representation to 3 continuous output values (popularity metrics).

## 📂 Project Structure

```bash
MoFA-Lite/
├── Data/                   # Dataset directory
│   ├── train/              # Training features (.npy)
│   └── test/               # Test features (.npy)
├── models/                 # Saved model checkpoints
│   ├── improved_10fold/    # Weights for each fold
│   └── scalers/            # Preprocessing scalers (Pickle)
├── feature_analysis/       # Generated analysis plots
├── Feature_Analysis.py     # Script for data exploration
├── train.py                # Main training script
├── test.py                 # Inference/Prediction script
├── README.md               # You are here!
└── LICENSE
```

## 🛠️ Getting Started

### Prerequisites

Ensure you have Python 3.8+ and PyTorch installed. You will also need `scikit-learn`, `numpy`, and `matplotlib`.

```bash
pip install torch numpy scikit-learn matplotlib
```

### 1. Data Preparation & Analysis
Place feature files (`.npy`) in the `Data/train` and `Data/test` directories as shown in the structure above.

To understand the data distribution and visualize feature correlations, run the following script:
```bash
python Feature_Analysis.py
```
> **Acknowledgement**: Special thanks to [**Peng Wang**](https://github.com/Unijockey) for his valuable insights into the input feature characteristics, which significantly guided our preprocessing strategy and led to substantial performance improvements.
### 2. Training
Run the training script to start the 10-Fold Cross-Validation process. This will train 10 separate models and save the best weights for each fold.

```bash
python train.py
```
*Key configurations (like `BATCH_SIZE`, `LR`, `HIDDEN_DIM`) can be modified directly in `train.py`.*

### 3. Inference
Once training is complete, generate predictions on the test set. This script loads all 10 models, performs inference, and averages the results for maximum robustness.

```bash
python test.py
```
The output will be saved as `predictions.npy`.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Citation

If you find this code useful for your research or project, please consider giving it a star ⭐ and citing it in your Final Examination paper:

```bibtex
@misc{MoFA-Lite2025,
  author = {Jiaxin Chen},
  title = {MoFA-Lite: Lightweight Modality-Focused Attention for Multimodal Video Popularity Prediction},
  year = {2025},
  teacher = {Zhenzhong Chen, Daiqin Yang},
  course = {Artificial Intelligence and Machine Learning},
  howpublished = {\url{https://github.com/PlutoKirito/MoFA-Lite}}
}
```

If you have any questions about this repo, feel free to contact me: kiritojiaxin@whu.edu.cn

---
*Built with ❤️*
