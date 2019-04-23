# For JSD and WD estimates:

JSD: `python jsd_estimate.py`
WD: `python wd_estimate.py`

The required plots will be saved as `js_estimate.png` and `wd_estimate.png` respectively.

# For density estimation:

Run `python density_estimation.py`. The actual distributtion of point will be saved in `exact.png` and the estimate using optimal discriminator will be saved in `estimate.png`.

# For Binary VAE:

python >= 3.63
pytorch >= 1.00

# Command to run:

mkdir -p saved_params

python BinaryVAE.py --epochs 20 --learning-rate -4 --batch-size 64 --imp-samples 200

For VAE:
python vae_clean.py --n_epochs 400 --batch_size 64 --optim_lr 2e-4

# For GAN

For training:
python gan_train_svhn_clean.py 

For generating, interpolation and disentanglement
python gan_generate_svhn_clean.py
