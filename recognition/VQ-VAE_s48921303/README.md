# VQ-VAE for Generating Privacy-Preserving Synthetic HipMRI cohorts

## Feasibility Review
## Project Description
This project involves the generation of synthetic HipMRI medical cohorts. These generated HipMRI cohorts will be used for data augmentation and anonymised sharing. The objective is to generate authentic physiological tissue boundaries and structures while abiding by the stringent patient data privacy laws.

### User Need, Scope & Acceptance Criteria
Synthesizing medical cohorts that preserve authentic HipMRI scans while adhering to patient confidentiality law, such as HIPAA and GDPR, expands research possibilities beyond scare clinical datasets. If successful, researchers can freely train and share models without risking patient confidentiality and the amount of data would dramatically increase.

Testable acceptance criteria:
1. Structural similarity index measure (SSIM) > 0.6
1. other testable criteria? do i need to justify?
1. what operational/resource constraint? do i need to justify this as well?

### Model Choice & Course Concepts
The proposed dataset is HipMRI 2D sliced data which are split into train, test and validation data. Each patient can only be in one of these 3 splits. If necessary, input slices can be padded with 0s. This may be desired because the data has a mix of 1 x 256 x 144 and 1 x 256 x 128 dimension slices. However, the proposed dataset will just dicard the slices of dimensions 1 x 256 x 144 because there are very few number of such slices.

The proposed model architecture is the VQ-VAE (hard difficulty). VQ-VAE is an autoencoder that uses a discrete representation of latent space rather than continous space used in VAE [1]. By doing so, VQ-VAE avoids posterior collapse that VAE often suffers from due to powerful decoders that can ignore latent variables. Increasing latent space to avoid posterior collapse is difficult as all possible sample points have to generate valid images. VQ-VAE avoids this problem by freely mapping discrete latent codes so it is not forced to match its latent representation to a predefined continuous prior. 

Furthermore, discrete representations are a more natural fit to HipMRI scans as these scans are composed of sharp distinct bones and tissue [1]. It does not make sense anatomically to have intermediate states between HipMRI scans that a VAE would have which arises from the continous representation of latent space.

Additionally, unlike VQ-VAE, VAE adds stochastic noise in every forward pass. This results in large variance which can cause unstable training [1]. Combined with the effective use of the latent space, VQ-VAE models can successfully model high-dimensional data rather than needing to spend computational resources on noise [1].

how much detail in explanation of what vq-vae is in feasibility report?
what is considered a baseline model? how much effort and time should i be putting into the architecture? what is the computational resource limit on this baseline?

### Preliminary Feasibility Evidence
The dataset consists of slices of HipMRI scans stored in NifTI format which are split into train, test and validation data. Each patient cannot be in more than one of these 3 splits. The train data consists of 11400 scans of shape (256, 128) and 60 of shape (256, 144). The test data contains of 540 scans of shape (256, 128). The test data consists of 660 scans of shape (256, 128). The scans are greyscale so their channel dimensions is 1.

## Artificial Intelligence Usage Disclosure
Tools used:
- Google Gemini

Tasks AI assisted with:
1. Got the functions and their specifics (such as what is actually tracked and units) needed to measure peak VRAM because I could not find any documentation via googling. Verified by reading the documentation and testing the functions.

## References
[1] https://arxiv.org/pdf/1711.00937