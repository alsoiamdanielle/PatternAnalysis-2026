# notes for self for what im going to write in readme.md
- when explaining vqvae need to highlight key things such as discrete  latent variables
- baseline model (vae) should be  matching architecture of vqvae for comparison and then what the vqvae provides that vae doesnt rather than just using baseline in paper

# VQ-VAE for Generating Privacy-Preserving Synthetic Hip MRI Cohorts

## Project Description
In this project, I aim to address the challenges of clinical data scarcity and stringent patient privacy laws by using a Vector Quantised Variational Autoencoder (VQ-VAE) to synthesise realistic 2D hip MRI cohorts. Due to strict patient confidentiality laws, medical imaging is frequently restricted from open research, leading to limited clinical datasets. This constrains the development of robust models and clinical assessment tools. To overcome these limitations, I developed a generative modelling pipeline pairing a VQ-VAE with a Gated PixelCNN prior, benchmarked against a baseline Variational Autoencoder (VAE). The primary objective is to evaluate whether discrete representation learning can faithfully capture authentic physiological tissue boundaries and anatomical structures in hip MRI scans. By generatingr realistic such realistic MRI cohorts that preserve patient privacy, I aim to expand scarce medical imaging datasets and facilitate data augmentation for downstream research.

## Models
### VQ-VAE
A VQ-VAE is a type of autoencoder that maps high-dimensional data into a discrete latent space. The model first compresses the input through an encoder into continuous spatial feature maps. Then, these features are quantised by mapping each vector to its nearest neighbour in a learned discrete codebook before a decoder reconstructs the input from these quantised features. This structure is shown on the left in Figure 1 for an image. The output of the encoder $z_e(x)$ is mapped to the nearest point $e_i$ for some $1 \le i \le k$. The right of Figure 1 shows the embedding space for the codebook, showing the assignment of $z_e(x)$ to $e_i$ that minimises the Euclidean distance.

![Figure 1: Architecture of the VQ-VAE [1]](README_images/VQ_VAE_Architecture.png)
Figure 1: Architecture of the VQ-VAE [1]

The fundamental difference between a VQ-VAE and a VAE is this latent space. The VAE's continuous Gaussian sampling is replaced with this deterministic vector quantisation step. This largely mitigates the posterior collapse that often occurs with VAEs as the model is not forced into an often uninformative Gaussian prior [1]. Furthermore, the discrete representation of the latent space means there is no continuous blending which would produce impossible interpolation artifacts. This is extremely important for the generation of hip MRI scans as valid anatomical structures are discrete and intermediate states between them do not exist in reality.

### PixelCNN
Because the VQ-VAE only compresses the input into a discrete grid of codebook indices, an autoregressive model is required to learn the distribution over these latent tokens to synthesise new data. Following the original paper [1], I use a PixelCNN to model this prior.

A PixelCNN is an autoregressive model that facotorises the joint probability of an image into a product of conditional probabilities [2]. The probability of an element is conditioned on all the previously generated elements, as visualised on the left of Figure 2. In training, this is achieved by masking a convolutional kernel to prevent future spatial information from leaking into the current prediction as shown in the centre of Figure 2. However, this masking creates the blind spot seen in the top right of Figure 2. Because the mask zeroes out all weights to the right of the centre pixel at every layer, information can never propagate rightward through intermediate layers, creating this blind spot.

To resolve this, Gated PixelCNN splits the architecture into a vertical stack which conditions on all rows above and a horizontal stack which conditions only on the previous pixels in the current row. This eliminates the blind spot without leaking future spatial information as shown in the bottom right of Figure 2 [2].

![Figure 2: Architecture of the PixelCNN [2]](README_images/PixelCNN_Architecture.png)
Figure 2: Architecture of the PixelCNN [2]

## Feasibility Review
### User Need, Scope & Acceptance Criteria
Synthesising medical cohorts that preserve authentic Hip MRI scans while adhering to patient confidentiality laws, such as HIPAA and GDPR, expands clinical research opportunities beyond scarce medical datasets. By generating realistic synthetic scans, researchers can openly train and distribute diagnostic models without compromising patient privacy, significantly increasing these scarce datasets.

Testable acceptance criteria:
1. Structural similarity index measure (SSIM) > 0.6: Measures structural fidelity rather than relying only on raw pixel differences.
1. 15 < Fréchet Inception Distance (FID) < 50: Measures distribution fidelity and is bounded to balance the generation of authentic images against training set memorisation to ensure plausible feature distributions without violating privacy laws.
1. Codebook utilisation > 0.6 (normalised codebook perplexity): Checks for codebook collapse to ensure that the majority of available discrete embedding vectors are contributing to the representations.
1. Batch inference latency < 50 ms (Batch Size = 32): Checks that synthetic batch sampling on an NVIDIA GeForce RTX 3090 executes quickly enough for deployment.
1. Peak training VRAM < 500 MB: Checks if the model maintains a light peak inference VRAM.

### Model Choice & Course Concepts
The target dataset consists of 2D slices from Hip MRI scans, split into training, validation, and testing subsets. Slices are split at the patient level so that no patient appears in more than one split, preventing data leakage. The raw scans contain a mix of $1 \times 256 \times 144$ and $1 \times 256 \times 128$ slices with only 60 out of the 12 600 slices having dimensions of $1 \times 256 \times 144$. The proposed dataset configuration involves discarding these 60 slices to avoid introducing zero-padding artifacts.

The proposed model architecture is the VQ-VAE (hard difficulty). The VQ-VAE replaces the continuous latent space of standard VAEs with a learned, discrete codebook. This deterministic vector quantisation step mitigates posterior collapse, an issue common to VAEs when powerful decoders learn to ignore continuous latent variables. While mitigating collapse in VAEs by expanding continuous latent dimensions requires ensuring every sampled point maps to a valid output, VQ-VAEs avoid this by mapping directly to discrete codes without being bound to a static Gaussian prior [1].

Furthermore, discrete representations provide a natural fit for Hip MRI data as these scans consist of distinct anatomical structures [1]. Anatomically, intermediate interpolations do not exist in reality. Continuous latent spaces blend these boundaries into non-physical artifacts. Additionally, VAEs inject stochastic sampling noise during every forward pass, potentially causing training instability [1]. By removing this continuous noise injection, the VQ-VAE dedicates its capacity entirely to encoding spatial features [1].

Because a VQ-VAE maps images deterministically to discrete indices, it cannot independently synthesise images via simple random sampling. A second model is required to learn the prior probability distribution over this discrete latent grid [1]. Following the original VQ-VAE architecture, the autoregressive PixelCNN is used as this prior [1]. Once trained, the PixelCNN autoregressively samples configurations of codebook indices, which are then passed to the VQ-VAE's decoder to generate synthetic Hip MRI slices.

### Preliminary Feasibility Evidence
The dataset comprises 2D Hip MRI slices stored in NIfTI format, strictly isolated by patient across splits. The training set consists of 11 400 scans of shape $(256, 128)$ and 60 scans of shape $(256, 144)$ (to be discarded). The validation set consists of 540 scans of shape $(256, 128)$, and the test set consists of 660 scans of shape $(256, 128)$. All scans are greyscale, resulting in a single channel dimensions.

I ran a lightweight VAE with 2 convolutional layers each for the encoder and decoder with 16 latent variables on an NVIDIA GeForce RTX 3090 as the baseline smoke-test experiment:
| Epoch | Loss (MSE + KL) | Validation SSIM | Epoch Time (s) | Cumulative Time (s) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2 638.65 | 0.1419 | 4.68 | 4.89 | 338.56 |
| **2** | 840.47 | 0.2144 | 2.78 | 7.75 | 338.56 |
| **3** | 654.57 | 0.2485 | 2.79 | 10.63 | 338.56 |
| **4** | 578.11 | 0.2685 | 2.77 | 13.48 | 338.56 |
| **5** | 538.26 | 0.2979 | 2.82 | 16.39 | 338.57 |
| **6** | 513.05 | 0.3020 | 2.85 | 19.34 | 338.57 |
| **7** | 496.40 | 0.3285 | 2.66 | 22.07 | 338.57 |
| **8** | 479.66 | 0.3061 | 2.42 | 24.56 | 338.57 |
| **9** | 465.73 | 0.3272 | 2.41 | 27.04 | 338.58 |
| **10** | 455.29 | 0.3376 | 2.40 | 29.51 | 338.58 |

This baseline confirms computational feasibility as the unoptimised, low-capacity VAE achieved a validation SSIM of 0.3376 while consuming under 340 MB of peak VRAM and completing 10 epochs in under 30 seconds (including logging and resource tracking overhead). While loss and SSIM plateau in later epochs, this stagnation stems directly from the restricted capacity of this VAE.

### Risks, Budget & Fallback
My next planned experiments involve implementing and tuning the full VQ-VAE and PixelCNN pipeline to satisfy the acceptance criteria. Key tuning ideas include:
- Architecture: Number of layers, channel dimensions, kernel size, stride, padding and use of skip connections
- Regularisation and activations: normalisation (e.g. BatchNorm) and activation functions
- Loss: Reconstruction loss, commitment loss hyperparameter
- Codebook hyperparameters: Number of embeddings and embedding dimensions
- Evaulation: Data range for SSIM, Inception-v3 feature layer (FID), sample sizes for running inference

The computational budget will significantly exceed the baseline smoke test due to increased architectural capacity and the two-stage training requirement:
1. VQ-VAE training: Scaling network depth and extending training duration is projected to take around 2 hours with under 5GB of peak VRAM.
1. PixelCNN training: Training is efficient as masking of the kernel enables parallelism. Furthermore, PixelCNN relies on downsampled discrete latent indices rather than the full image. As a result, training is projected to take around 30 minutes with under 2 GB of peak VRAM.

If the VQ-VAE fails to reach target SSIM, I will expand the model capacity by scaling network depth, codebook dimensions and latent feature grid. This may come at the cost of allowing memorisation of training data.

## Reproducibility
### Software Dependencies
- Python: 3.12.10
- PyTorch (torch): 2.14.1
- Torchvision (torchvision): 0.29.1
- TorchMetrics (torchmetrics): 1.9.0
- NiBabel (nibabel): 5.4.2
- NumPy (numpy): 2.3.4
- Matplotlib (matplotlib): 3.10.7

### Hardware
- GPU: NVIDIA GeForce RTX 3090 with 24 GB GDDR6X VRAM
- 24 GB of VRAM accommodated full batch training (batch size of 64)
- GPU memory consumption measured within train.py using PyTorch CUDA runtime APIs.

### Execution
Execution involves a two-stage pipeline where `train.py` must be executed first followed by `predict.py`. `train.py` trains the models, tracks validation convergence, and writes the timestamped weights and training curves to disk. `predict.py` loads the saved weights to perform final test evaluation, **??**

#### 1. Model Training (`train.py`)

`train.py` handles data loading with training-only data augmentation, optimiser setup, validation metric tracking, early stopping, and automatic logging of model weights, CSV metrics, and training curves.

##### CLI Arguments for `train.py`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | Required | Architecture to train: `'vqvae'` or `'vae'`. |
| `--data_dir` | `str` | Required | Root directory containing `keras_slices_train` and `keras_slices_validate`. |
| `--vqvae_epochs` | `int` | `30` | Maximum training epochs for the VQ-VAE model (monitored by early stopping). |
| `--pixelcnn_epochs`| `int` | `60` | Training epochs for the PixelCNN. |
| `--vae_epochs` | `int` | `30` | Maximum training epochs for the baseline VAE model. |
| `--batch_size` | `int` | `64` | Mini-batch size for DataLoader instances. |
| `--num_embeddings` | `int` | `32` | Codebook dictionary size ($K$) for VQ-VAE and token vocabulary for PixelCNN. |
| `--embedding_dim` | `int` | `64` | Channel dimensionality ($D$) of codebook vectors and latent feature maps. |
| `--lr` | `float` | `0.0001` | Learning rate for the Adam optimiser. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--test` | `flag` | `False` | Truncates dataset loading to 20 slices for pipeline smoke-testing. |

##### Example Training Commands
Train the baseline VAE model:
```
python train.py --model vae --data_dir ./HipMRI --embedding_dim 64 --batch_size 64 --vae_epochs 30 --lr 0.0001
```

Train the VQ-VAE + PixelCNN pipeline:
```
python train.py --model vqvae --data_dir ./HipMRI --num_embeddings 32 --embedding_dim 64 --batch_size 64 --vqvae_epochs 30 --pixelcnn_epochs 60 --lr 0.0001
```

#### 2. Model Evaluation (`predict.py`)

## Preprocessing
### Data Preprocessing
- Loading Dataset: Hip 2D slices are loaded from NIfTI (`.nii`) using `nibabel`. Any extra singleton or fourth dimensions present are removed so that inputs are all 2D single-channel matrices.
- Standardisation And Filtering: The dataset consists almost entirely of $256 \times 128$ slices, but contains 60 slices of shape $256 \times 144$ out of the 12,600 total scans. To avoid introducing zero-padding artifacts, boundary distortions, or spatial resizing, these 60 slices are discarded from the training set.
- Normalisation: To stabilise loss gradients and match the decoder's final `Sigmoid()` activation layer, each slice is scaled to the range $[0.0, 1.0]$ using slice min-max normalisation.
- Data Augmentation: To improve generalisation while preserving anatomical structures, data augmentations are applied on to the training (`augment=True`):
  - `RandomHorizontalFlip(p=0.5)`: Horizontal flip 50% of the time
  - `RandomAffine(degrees=5, translate=(0.02, 0.02))`: Applies slight rotations and translations to simulate slight positioning variations on the MRI scanner.

### Data Splitting
- Patient-Level Segregation: Slices are partitioned at the patient level. This ensures slices from the same patient do not appear across multiple splits, preventing data leakage.
- Splits:
  - Training Set (`keras_slices_train`): 11,400 slices of shape $256 \times 128$ (after discarding the 60 outlier slices). Used to train the VQ-VAE and PixelCNN.
  - Validation Set (`keras_slices_validate`): 540 slices of shape $256 \times 128$. Evaluated at the end of each epoch to track validation SSIM and codebook perplexity, using early stopping (`patience=7`) to restore the best model weights.
  - Test Set (`keras_slices_test`): 660 slices of shape $256 \times 128$. Reserved for `predict.py` to evaluate test metrics (SSIM, FID, KID), **??**

## Training
### Hyperparameters Used
#### Training & Optimization Hyperparameters
- Optimiser: Adam (`torch.optim.Adam`)
- Learning Rate (`--lr`): 0.0001 (1e-4)
- Training Batch Size (`--batch_size`): 64
- Evaluation Batch Size: 32
- Random Seed (`--seed`): 42
- Early Stopping Patience: 7 epochs (monitored on validation SSIM for both VQ-VAE and VAE)

#### Data & Preprocessing Hyperparameters
- Image Input Dimensions: 256 x 128 x 1 (single-channel greyscale)
- Intensity Scaling: Per-slice min-max normalisation to [0.0, 1.0]
- Data Augmentations (Training Only):
  - Horizontal Flip: `RandomHorizontalFlip(p=0.5)`
  - Spatial Jitter: `RandomAffine(degrees=5, translate=(0.02, 0.02))`

#### VQ-VAE Hyperparameters
- Maximum Epochs (`--vqvae_epochs`): 30
- Codebook Size (K, `--num_embeddings`): 32 embeddings
- Embedding Dimension (D, `--embedding_dim`): 64 channels
- Latent Feature Map Resolution: 64 x 32 (downsampled by a spatial factor of 4)

#### Gated PixelCNN Prior Hyperparameters
- Training Epochs (`--pixelcnn_epochs`): 60
- Token Vocabulary Size (`num_embeddings`): 32 (matches VQ-VAE codebook size K)
- Gradient Clipping Norm: max_norm = 1.0

#### Baseline VAE Hyperparameters
- Maximum Epochs (`--vae_epochs`): 30
- Latent Channel Dimension (`--embedding_dim`): 64
- KL Divergence Weight ($\beta$): 1.0 (standard ELBO formulation)

### Training Metric Plots
![Figure 3: VQ-VAE Training Loss](README_images/VQ_VAE_Loss_Curve.png)

Figure 3: VQ-VAE Training Loss

This plot tracks total training loss per epoch, combining mean squared reconstruction error and vector quantisation commitment loss. The loss drops sharply from 5.41 at epoch 1 to under 0.5 by epoch 3, before smoothly flattening towards 0.03 by epoch 30 without divergence or loss spikes. This validates stable optimisation under the straight-through estimator and Adam learning rate (0.0001), demonstrating computational convergence well within the projected two-hour training budget from the feasibility report.

![Figure 4: VQ-VAE Validation SSIM](README_images/VQ_VAE_SSIM_Curve.png)

Figure 4: VQ-VAE Validation SSIM

This curve measures structural reconstruction fidelity across validation slices at the end of each epoch. The validation SSIM climbs steadily from 0.29 at epoch 1, passes 0.6 by epoch 8, and converges at a peak of 0.765 by epoch 30. This directly satisfies and exceeds Acceptance Criterion 1 (Validation SSIM > 0.6) from the feasibility review, confirming that discrete latent quantisation faithfully captures the hip MRI scans.

![Figure 5: VQ-VAE Codebook Perplexity](README_images/VQ_VAE_Perplexity_Curve.png)

Figure 5: VQ-VAE Codebook Perplexity

This plot tracks normalised codebook perplexity, measuring the active utilisation and distribution uniformity of the 32 discrete codebook vectors on validation data. Perplexity increases from 0.15 at initialisation, peaks at 0.54 around epoch 6, and stabilises into a steady plateau between 0.46 and 0.48 across epochs 15 to 30. Although it settles slightly below the target of 0.60 set in Acceptance Criterion 3, the steady plateau confirms healthy codebook participation without codebook collapse, potentially showing that hip MRI scans naturally concentrates into a compact subset of discrete tokens.

![Figure 6: PixelCNN Training Loss](README_images/PixelCNN_Loss_Curve.png)

Figure 6: PixelCNN Training Loss

This curve displays the cross-entropy training loss per epoch for the Gated PixelCNN prior learning the distribution over the latent codebook grid. Cross-entropy decreases quickly from 1.71 to 1.34 over the first 5 epochs and steadily decays to 1.14 by epoch 60 without oscillation. While the loss curve has not fully flattened into a plateau indicating that complete convergence has not yet been reached, 60 epochs requires substantial computational resources where further training offers diminishing returns.

![Figure 7: VAE Training Loss](README_images/VAE_Loss_Curve.png)

Figure 7: VAE Training Loss

This plot shows the baseline VAE training loss, computed as the sum of reconstruction MSE and KL divergence across 30 epochs. The loss falls steeply from 2 750 down to approximately 530 within 4 epochs, before plateauing near 400 from epoch 10 and onwards. This curve the matches what is seen in the preliminary feasibility smoke test.

![Figure 8: VAE Validation SSIM](README_images/VAE_SSIM_Curve.png)

Figure 8: VAE Validation SSIM

This curve tracks validation SSIM per epoch for the baseline VAE. The metric rises rapidly from 0.13 to around 0.33 by epoch 5, but slows noticeably around epoch 15 and caps out at 0.407 by epoch 30 with considerable fluctuations across epochs. This fails the project acceptance threshold of SSIM > 0.6. Benchmarking this 0.407 ceiling against the VQ-VAE's 0.765 illustrates that the VQ-VAE achieves significantly superior reconstruction fidelity.

### Input and Reconstruction

### Hip MRI Generation

## Investigation of the Open Research Dilemma
### Quantitative Benchmarking Against Baseline

### Computational Resource Profiling

### Qualitative Error Autopsies

### Engineering Recommendations and Trade-Off Synthesis

## Artificial Intelligence Usage Disclosure
### Tools Used
- Google Gemini

### Tasks AI Assisted With:
1. Got the functions and their specifics (such as what is actually tracked and units) needed to measure peak VRAM because I could not find any documentation via googling. Verified by reading the documentation and testing the functions.
1. Formatting my data to give the table under Preliminary Feasibility Evidence. Verified by simply viewing the output.
1. Got a list of ideas for testable acceptance criteria. I didn't just ask AI for list I gave it ideas for what to measure and then went through all criteria to pick based on my mine evaluation. Evaulated of recommended thresholds were reasonable by testing against code.
1. Explained to me how PixelCNN works. I did not directly use anything from the AI so no validation of code or text is needed.
1. AI gave me projected time and peak VRAM for project. What I ultimately wrote in the feasbility report was using my own thoughts and evaluating the AI's reasoning for the projected numbers.
1. AI assisted in the implementation for PixelCNN. I didn't just blindly copy in code, rather I consulted AI about the implementation. This way, I decided what to include and change based on such consultations by evaluating what the AI had to say.
1. AI assisted in adjusting pixelCNN architecture and hyperparameters for generation. Again, as the point above, I didn't just blindly copy code.
1. AI assisted in implementing measurements of distribution fidelity (FID and KID). Again, as the point above, I didn't just blindly copy code.
1. AI assisted with wording of certain ideas I had for writing documentation in README. I always read what AI came up with and then adapted it to my liking without blinding copy and pasting.
1. AI assisted with cleaning up of functionaly complete train.py and predict.py such helping with parsing arguments, plotting, getting codebook perplexity and etc. Again, I didn't just blindly copy code.

## References
[1] https://arxiv.org/pdf/1711.00937 - vqvae

[2] https://arxiv.org/pdf/1606.05328 - pixelcnn