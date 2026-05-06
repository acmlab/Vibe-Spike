python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# install torch with your CUDA version, for example, if you have CUDA 12.8, use the following command:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt