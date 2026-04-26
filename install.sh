# Prerequisites:
#
# 1. Install CUDA 11.8
#    Follow intructions on https://developer.nvidia.com/cuda-11-8-0-download-archive
#    Make sure that
#      -   PATH includes <CUDA_DIR>/bin
#      -   LD_LIBRARY_PATH includes <CUDA_DIR>/lib64
#    If needed, restart bash environment

#    The environment was tested only with this CUDA version

# 2. Install Blender 3.6 to create strand visualizations
#    Follow instructions on https://www.blender.org/download/lts/3-6
#

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

# Save parent dir
PROJECT_DIR=$PWD

# Pull all external libraries
mkdir ext
cd $PROJECT_DIR/ext && git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose --depth 1
cd $PROJECT_DIR/ext/openpose && git submodule update --init --recursive --remote
cd $PROJECT_DIR/ext && git clone https://github.com/hustvl/Matte-Anything
cd $PROJECT_DIR/ext/Matte-Anything && git clone https://github.com/IDEA-Research/GroundingDINO.git
cd $PROJECT_DIR/ext && git clone git@github.com:egorzakharov/NeuralHaircut.git --recursive
cd $PROJECT_DIR/ext && git clone https://github.com/facebookresearch/pytorch3d
cd $PROJECT_DIR/ext/pytorch3d && git checkout 2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7
cd $PROJECT_DIR/ext && git clone https://github.com/camenduru/simple-knn
cd $PROJECT_DIR/ext/diff_gaussian_rasterization_hair/third_party && git clone https://github.com/g-truc/glm
cd $PROJECT_DIR/ext/diff_gaussian_rasterization_hair/third_party/glm && git checkout 5c46b9c07008ae65cb81ab79cd677ecc1934b903
cd $PROJECT_DIR/ext && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd $PROJECT_DIR/ext/kaolin && git checkout v0.15.0
cd $PROJECT_DIR/ext && git clone https://github.com/SSL92/hyperIQA

# Install environment
cd $PROJECT_DIR && conda env create -f environment.yml
conda activate gaussian_splatting_hair
pip install ext/pytorch3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install ext/NeuralHaircut/npbgpp
pip install ext/simple-knn
pip install ext/diff_gaussian_rasterization_hair
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html

# Download Neural Haircut files
cd $PROJECT_DIR/ext/NeuralHaircut
gdown --folder https://drive.google.com/drive/folders/1TCdJ0CKR3Q6LviovndOkJaKm8S1T9F_8
cd $PROJECT_DIR/ext/NeuralHaircut/pretrained_models/diffusion_prior # downloads updated diffusion prior
gdown 1_9EOUXHayKiGH5nkrayncln3d6m1uV7f
cd $PROJECT_DIR/ext/NeuralHaircut/PIXIE
gdown 1mPcGu62YPc4MdkT8FFiOCP629xsENHZf && tar -xvzf pixie_data.tar.gz ./ && rm pixie_data.tar.gz
cd $PROJECT_DIR/ext/hyperIQA && mkdir pretrained && cd pretrained
gdown 1OOUmnbvpGea0LIGpIWEbOyxfWx6UCiiE
cd $PROJECT_DIR

# Matte-Anything
conda create -y -n matte_anything python=3.9 pip=23.3.1
conda activate matte_anything
conda install -y --strict-channel-priority pytorch::pytorch=2.0.0 pytorch::torchvision=0.15.0 pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorboard timm==0.5.4 opencv-python mkl==2024.0 setuptools easydict wget scikit-image fairscale
pip install git+https://github.com/facebookresearch/segment-anything.git
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
cd $PROJECT_DIR/ext/Matte-Anything/GroundingDINO && pip install --no-build-isolation -e .
pip install supervision==0.22.0 # fixes the GroundingDINO error
cd $PROJECT_DIR/ext/Matte-Anything && mkdir pretrained
cd $PROJECT_DIR/ext/Matte-Anything/pretrained
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
conda deactivate && conda activate gaussian_splatting_hair
gdown 1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW

# OpenPose
cd $PROJECT_DIR/ext/openpose
gdown 1Yn03cKKfVOq4qXmgBMQD20UMRRRkd_tV && tar -xvzf models.tar.gz && rm models.tar.gz # downloads openpose checkpoint
conda deactivate
git submodule update --init --recursive --remote
conda create -y -n openpose cmake=3.20 -c conda-forge # needed to avoid cmake complining error
conda activate openpose
sudo apt install libopencv-dev # installation instructions are from EasyMocap, in case of problems refer to the official OpenPose docs
sudo apt install protobuf-compiler libgoogle-glog-dev
sudo apt install libboost-all-dev libhdf5-dev libatlas-base-dev
mkdir build
cd build
cmake .. -DBUILD_PYTHON=true -DUSE_CUDNN=off
make -j8
conda deactivate

# PIXIE
cd $PROJECT_DIR/ext && git clone https://github.com/yfeng95/PIXIE
cd $PROJECT_DIR/ext/PIXIE
chmod +x fetch_model.sh && ./fetch_model.sh
conda create -y -n pixie-env python=3.8
conda activate pixie-env
conda install -y --strict-channel-priority pytorch::pytorch=2.0.0 pytorch::torchvision=0.15.0 pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia
pip install fvcore kornia matplotlib
pip install ninja
export MAX_JOBS=8
pip install $PROJECT_DIR/ext/pytorch3d -v
pip install pyyaml==5.4.1
pip install git+https://github.com/1adrianb/face-alignment.git@54623537fd9618ca7c15688fd85aba706ad92b59 # install this commit to avoid error