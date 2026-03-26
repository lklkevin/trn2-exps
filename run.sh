# Configure Linux for Neuron repository updates
. /etc/os-release
cd "$(dirname "$0")"
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
sudo apt-get update -y

# Install OS headers
sudo apt-get install linux-headers-$(uname -r) -y

# Install git
sudo apt-get install git -y

# Install Neuron Driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Runtime
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools
sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH

# Install Python venv
sudo apt-get install -y python3.12-venv g++

# Create Python venv
python3.12 -m venv aws_neuron_venv_pytorch

# Activate Python venv
source aws_neuron_venv_pytorch/bin/activate
python -m pip install -U pip

# Install Jupyter notebook kernel
pip install ipykernel
python3.12 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli
python -m pip install wget
python -m pip install awscli

# Install Neuron Compiler and Framework
python -m pip install neuronx-cc==2.* torch-neuronx==2.9.* torchvision nki

python experiments/zero_skip_benchmark.py --mode benchmark --output results/zero_skip_results.json --variants load_skip_matmul,skip_nostore
python experiments/bucket_sweep.py --mode benchmark --dim m --output results/bucket_sweep_m.json
python analysis/plot_zero_skip.py results/zero_skip_results.json --output-dir results/plots
python analysis/plot_bucket_sweep.py results/bucket_sweep_m.json --output-dir results/plots