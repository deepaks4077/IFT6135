# For Binary VAE:

python >= 3.63
pytorch >= 1.00

# Command to run:

mkdir -p saved_params

python BinaryVAE.py --epochs 20 --learning-rate -4 --batch-size 64 --imp-samples 200
