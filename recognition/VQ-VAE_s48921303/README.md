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
| **1** | 2,638.65 | 0.1419 | 4.68 | 4.89 | 338.56 |
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

### Hardware

### Execution

## Preprocessing
### Data Preprocessing

### Data Splitting

## Example Inputs, Outputs, and Visual Plots
### Training Metric Plots

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