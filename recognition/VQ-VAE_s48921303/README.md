# VQ-VAE for Generating Privacy-Preserving Synthetic HipMRI cohorts

## Project Description
This project involves the generation of synthetic HipMRI medical cohorts. These generated HipMRI cohorts will be used for data augmentation and anonymised sharing. The objective is to generate authentic physiological tissue boundaries and structures while abiding by the stringent patient data privacy laws.

## Feasibility Review
### User Need, Scope & Acceptance Criteria
Synthesising medical cohorts that preserve authentic HipMRI scans while adhering to patient confidentiality law, such as HIPAA and GDPR, expands research possibilities beyond scare clinical datasets. If successful, researchers can freely train and share models without risking patient confidentiality and the amount of data would dramatically increase.

Testable acceptance criteria:
1. Structural similarity index measure (SSIM) > 0.6 - measures strucutral slimilarity rather than just raw pixel differences
1. 15 < Fréchet Inception Distance (FID) < 50 - measures distribution fidelity and is bounded to balance generation of authentic images against training set memorisation, ensuring plausible feature distributions without violation of privacy laws.
1. Code utilisation > 0.6 measured with codebook perpexity - checks for codebook collapse where only some codebook vectors are used
1. Synthesising 32 (batch) images < 50ms on NVIDIA GeForce RTX 3090 - checks if inference in a reasonably short amount of time for deployment
1. Peak training VRAM of < 500MB - checks if VRAM used is resonable for deployment

### Model Choice & Course Concepts
The proposed dataset is HipMRI 2D sliced data which are split into train, test and validation data. Each patient can only be in one of these 3 splits. If necessary, input slices can be padded with 0s. This may be desired because the data has a mix of 1 x 256 x 144 and 1 x 256 x 128 dimension slices. However, the proposed dataset will just dicard the slices of dimensions 1 x 256 x 144 because there are very few number of such slices.

The proposed model architecture is the VQ-VAE (hard difficulty). VQ-VAE is an autoencoder that uses a discrete representation of latent space rather than continous space used in VAE [1]. By doing so, VQ-VAE avoids posterior collapse that VAE often suffers from due to powerful decoders that can ignore latent variables. Increasing latent space to avoid posterior collapse is difficult as all possible sample points have to generate valid images. VQ-VAE avoids this problem by freely mapping discrete latent codes so it is not forced to match its latent representation to a predefined continuous prior. 

Furthermore, discrete representations are a more natural fit to HipMRI scans as these scans are composed of sharp distinct bones and tissue [1]. It does not make sense anatomically to have intermediate states between HipMRI scans that a VAE would have which arises from the continous representation of latent space.

Additionally, unlike VQ-VAE, VAE adds stochastic noise in every forward pass. This results in large variance which can cause unstable training [1]. Combined with the effective use of the latent space, VQ-VAE models can successfully model high-dimensional data rather than needing to spend computational resources on noise [1].

Due to the discrete representation of the latent space, VQ-VAEs alone are not sufficient to synthesise images. Unlike VAEs, VQ-VAEs cannot just simply sample from a continuous distribution to produce an image. To deal with this, another model is needed to learn the prior probability distribution over the discrete latent space [1]. Since model choices for this generative prior were not covered, I will follow the original VQ-VAE paper and use a PixelCNN [1]. Once the PixelCNN is trained, I can sample a valid configuration of discrete codebook indices from the learnt prior and pass them to the VQ-VAE decoder to synethesis the HipMRI slices.

### Preliminary Feasibility Evidence
The dataset consists of slices of HipMRI scans stored in NifTI format which are split into train, test and validation data. Each patient cannot be in more than one of these 3 splits. The train data consists of 11400 scans of shape (256, 128) and 60 of shape (256, 144). The test data contains of 540 scans of shape (256, 128). The test data consists of 660 scans of shape (256, 128). The scans are greyscale so their channel dimensions is 1.

I ran a VAE with 2 layers for encoder and decoder and 16 latent variables on an NVIDIA GeForce RTX 3090 as the baseline smoke-test experiment. Here is the data:
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

This baseline experiment demonstrates feasibility as the unoptimised and low-capacity VAE achived a validation SSIM greater than 0.33 using very low computational resources. The 10-epoch run required less than 340MB of peak VRAM and finished under 30 seconds. The 30 seconds included both training the VAE and tracking of resources. Although the loss and SSIM plateau towards the later epochs, this stagnation is a result of architectural limits of the VAE using only using 2 layers for encoder and decoder and 16 latent variables.

### Risks, Budget & Fallback
My next planned experiments involves implementing the VQ-VAE and tuning the architecture and hyperparameters to meet the acceptance criteria. This may involve tuning or choosing:
- Number of layers, channel dimensions, kernel size, stride and padding in encoder and decoder
- Normalisation of data and batch normalisation
- Activation function
- Reconstruction loss
- Hyperparameter for loss function which scales commitment loss
- SSIM hyperpameters such as data range, kernel size and standard kernel
- Number of generated samples against real slices for measuring max SSIM
- Number of sampling when running inference to check duration
- Number of embeddings and embedding dimensions of VQ-VAE
- Use of skip connections

The computing budget will be significantly greater than what is shown in the table under Preliminary Feasbility Evidence as the VAE used was very lightweight. Furthermore, VQ-VAE introduces discrete representation of the latent space and requires another model to synthesise images. For the training of the VQ-VAE, given the increase in epochs running an increased complexity and size of an architecture, convergence is predicted to take around 2 hours with peak VRAM of less than 5GB. For the PixelCNN, the training is fast because the kernel is masked to zero out future tokens as opposed to computing for token n by using the state of token 2 [2]. As a result, in addition to training without sequential steps, PixelCNN relies on downsampled discrete index rather than the full image itself. So, the peak VRAM usage will be around less than 2GB with training time taking about 30 minutes.

If the VQ-VAE fails to reach target SSIM due to incapcity of architecture, I will intentionally allow the VQ-VAE to have more capacity to memorise the training data. This can involve strategies such as increasing layers, increasing codebook dimensions and etc.

# notes for self for what im going to write in readme.md
- when explaining vqvae need to highlight key things such as discrete  latent variables
- baseline model (vae) should be  matching architecture of vqvae for comparison and then what the vqvae provides that vae doesnt rather than just using baseline in paper

## Artificial Intelligence Usage Disclosure
Tools used:
- Google Gemini

Tasks AI assisted with:
1. Got the functions and their specifics (such as what is actually tracked and units) needed to measure peak VRAM because I could not find any documentation via googling. Verified by reading the documentation and testing the functions.
1. Formatting my data to give the table under Preliminary Feasibility Evidence. Verified by simply viewing the output.
1. Got a list of ideas for testable acceptance criteria. I didn't just ask AI for list I gave it ideas for what to measure and then went through all criteria to pick based on my mine evaluation. Evaulated of recommended thresholds were reasonable by testing against code.
1. Explained to me how PixelCNN works. I did not directly use anything from the AI so no validation of code or text is needed.
1. AI gave me projected time and peak VRAM for project. What I ultimately wrote in the feasbility report was using my own thoughts and evaluating the AI's reasoning for the projected numbers.
1. AI assisted in the implementation for PixelCNN. I didn't just blindly copy in code, rather I consulted AI about the implementation. This way, I decided what to include and change based on such consultations by evaluating what the AI had to say.

## References
[1] https://arxiv.org/pdf/1711.00937 - vqvae
[2] https://arxiv.org/pdf/1606.05328 - pixelcnn