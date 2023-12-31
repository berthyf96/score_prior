# Score-Based Priors for Bayesian Inverse Imaging

[Webpage](http://imaging.cms.caltech.edu/score_prior/) | [PDF](https://arxiv.org/abs/2304.11751)

![figure showing posterior samples using score-based priors under different measurement settings](assets/teaser.png "Score-Based Priors")


**BibTeX:**
```
@inproceedings{feng2023score,
  title={Score-Based Diffusion Models as Principled Priors for Inverse Imaging},
  author={Feng, Berthy T and Smith, Jamie and Rubinstein, Michael and Chang, Huiwen and Bouman, Katherine L and Freeman, William T},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023},
  organization={IEEE}
}
```

This project turns score-based diffusion models into explicit priors for Bayesian inverse problems in imaging. A "score-based prior" allows us to model complex, data-driven posterior distributions while using established sampling and optimization methods (e.g., MCMC, variational inference) that require an explicit log-density function. In our work, we propose a variational-inference approach to posterior sampling, in addition to empirically validating the accuracy and variance of a score prior.

## Getting started
Clone the repo:
```
git clone https://github.com/berthyf96/score_prior
cd score_prior
git submodule add https://github.com/berthyf96/score_flow
```

Install dependencies (please open an issue if this does not work out-of-the-box):
```
sh conda.sh
```

## Example workflow
### Train score model
To train a score-based prior, you first need to train a score-based diffusion model on the desired dataset:
```
python train.py \
  --config configs/score_config.py \
  --workdir score_checkpoints/CELEBA_32 \
  --config.data.dataset CELEBA \
  --config.data.image_size 32 \
  --config.data.tfds_dir /tmp/tensorflow_datasets \
  --config.training.n_iters 1000000
```
Once trained, the score-based diffusion model with parameters $\theta$ represents the image prior $p_\theta$.

### Optimize DPI for posterior sampling
Often, our goal is to sample from a posterior $$p_\theta(\mathbf{x}\mid\mathbf{y})\propto p(\mathbf{y}\mid\mathbf{x})p_\theta(\mathbf{x}).$$ We use [Deep Probabilistic Imaging](https://github.com/HeSunPU/DPI) (DPI) for posterior sampling, which optimizes the parameters of a RealNVP to approximate the target posterior. Here is an example command to perform DPI optimization:
```
python train_dpi.py \
  --score_model_config configs/score_config.py \
  --config configs/dpi_config.py \
  --workdir dpi_checkpoints/Denoising_CELEBAxCELEBA_32 \
  --config.prob_flow.score_model_dir score_checkpoints/CELEBA_32/checkpoints/checkpoint_1000000 \
  --config.data.dataset CELEBA \
  --config.data.image_size 32 \
  --config.data.num_channels 3 \
  --config.likelihood.likelihood Denoising \
  --config.likelihood.noise_scale 0.2 \
  --config.training.batch_size 32 \
  --config.training.n_iters 10000 \
  --config.training.snapshot_freq 100 \
  --config.optim.learning_rate 2e-4 \
  --config.optim.grad_clip 1. \
  --score_model_config.training.sde vpsde
```
Replace `config.prob_flow.score_model_dir` with the path to your trained score model.
You can also change the forward model by changing `config.likelihood`.


## Customization
Datasets can be added in `score_flow/datasets.py`.
Forward models can be added in `forward_models.py`.
