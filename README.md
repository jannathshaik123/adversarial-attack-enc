
# Adversarial Attack and Defense on MNIST and FashionMNIST

## Project Overview

This project explores adversarial attacks and defenses on image classification models, specifically using the MNIST and FashionMNIST datasets. The main objectives are:
- Train a VGG16-like convolutional neural network (CNN) on MNIST and FashionMNIST for digit and fashion item recognition.
- Implement and evaluate adversarial attacks, specifically the FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) attacks.
- Develop and train a convolutional autoencoder as a defense mechanism to reconstruct and denoise adversarially perturbed images.
- Analyze and visualize training, attack, and defense results.

## Repository Structure

- `notebook.ipynb`: Main Jupyter Notebook containing the code for data loading, training, attacks, defense, and visualization.
- Model weights and checkpoints are saved under `model` and `defense` directories (paths are specified in the notebook).
- Datasets are managed and downloaded automatically via PyTorch's `torchvision.datasets`.

## Requirements

- Python 3.8+
- PyTorch (tested with 2.5.1+cu121)
- torchvision
- torchsummary
- matplotlib
- numpy
- pandas
- PIL

To install the requirements, run:

```bash
pip install torch torchvision torchsummary matplotlib numpy pandas pillow
```

## Explanation

### 1. Data Preparation
- MNIST and FashionMNIST datasets are downloaded and split into train, validation, and test sets.
- Images are resized and normalized for compatibility with VGG-like models.

### 2. Model Architecture
- **VGG16 Variant:** A custom VGG16-like CNN is implemented, supporting optional batch normalization, and trained for 10 epochs on each dataset.
- The model achieves high accuracy (>98% for MNIST, >90% for FashionMNIST).

### 3. Adversarial Attacks
- **FGSM Attack:** Generates adversarial examples by perturbing images in the direction of the gradient of the loss w.r.t the input.
- **PGD Attack:** An iterative, stronger attack that applies multiple small FGSM steps with clipping.

### 4. Defense Mechanism
- **Convolutional Autoencoder:** Trained to reconstruct clean images from adversarial examples, effectively reducing the impact of attacks.
- The autoencoder uses GELU activations and batch normalization for stability and performance.

### 5. Training and Evaluation
- Training/validation loss and accuracy curves are plotted for both classifiers and the autoencoder.
- Attack success and defense performance are visualized and printed.

## Results Summary

This project evaluates the robustness of models against adversarial attacks (FGSM and PGD) using the MNIST and Fashion-MNIST datasets. The table below summarizes the key metrics:

| Dataset         | Attack | Test Accuracy | Accuracy (w/o Def) | Avg. MSE   | Avg. PSNR | Avg. SSIM |
|-----------------|--------|--------------|--------------------|------------|-----------|-----------|
| MNIST           | FGSM   | 0.7494       | 0.2648             | 0.028629   | 15.99     | 0.6106    |
| MNIST           | PGD    | 0.5532       | 0.0418             | 0.010043   | 20.83     | 0.8238    |
| Fashion-MNIST   | FGSM   | 0.6526       | 0.1417             | 0.081761   | 11.23     | 0.2401    |
| Fashion-MNIST   | PGD    | 0.4793       | 0.2942             | 0.022153   | 16.96     | 0.6392    |

**Key Points:**
- Both FGSM and PGD attacks significantly reduce model accuracy, especially without defense mechanisms.
- PGD tends to be a stronger attack than FGSM, yielding lower accuracy but higher image similarity (SSIM) on MNIST.
- Fashion-MNIST models are more vulnerable to FGSM than MNIST, as indicated by higher MSE and lower SSIM.
- Metrics such as PSNR and SSIM provide insight into the perceptual quality of adversarial examples.

---

- The VGG16 variant achieves state-of-the-art accuracy on both MNIST and FashionMNIST.
- Under FGSM and PGD attacks, classification accuracy drops significantly, demonstrating the effectiveness of adversarial attacks.
- The convolutional autoencoder is able to partially recover clean images from adversarially attacked inputs, improving post-defense classification accuracy.
- Plots show clear trends in loss/accuracy and the effectiveness of the defense.

## Conclusion

This project demonstrates that:
- Deep networks, while powerful, are vulnerable to adversarial attacks.
- Attacks like FGSM and PGD can cause high-performing models to misclassify with imperceptible perturbations.
- A well-trained autoencoder can serve as an effective first line of defense, mitigating some adversarial effects.
- Robustness remains an open challenge, and combining multiple defenses or using robust training may yield further improvements.

## Usage

1. Clone the repository and install the requirements.
2. Run `notebook.ipynb` in Jupyter Notebook.
3. Model files and results will be saved as specified in the notebook.
4. Modify paths as necessary for your local setup.
