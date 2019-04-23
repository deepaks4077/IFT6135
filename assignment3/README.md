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
