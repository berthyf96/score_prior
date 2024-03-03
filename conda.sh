# TensorFlow 2.12, CUDA 11.8, CuDNN 8.6
conda install -y pip
pip install --upgrade pip

# Install TensorFlow 2.12 and its dependencies
conda install -c conda-forge -y cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Other packages
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm
pip install jupyterlab
pip install matplotlib
pip install seaborn
git clone https://github.com/achael/eht-imaging.git /tmp/eht-imaging
pip install /tmp/eht-imaging
git clone https://github.com/liamedeiros/ehtplot /tmp/ehtplot
pip install /tmp/ehtplot
pip install diffrax
pip install flax
pip install ml-collections
pip install tensorflow-datasets
pip install scikit-learn
conda install -c conda-forge -y pynfft
# eht-imaging needs numpy <= 1.23
pip uninstall numpy
pip install numpy==1.23.*
pip install tensorflow-probability==0.20
pip install parameterized