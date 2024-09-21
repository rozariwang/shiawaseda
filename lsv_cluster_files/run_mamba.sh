nvcc --version || echo "nvcc not found"
ldconfig -p | grep cuda || echo "CUDA libraries not found"
nvidia-smi || echo "nvidia-smi not found"
python3 --version
python /nethome/hhwang/hhwang/shiawaseda/M2_BPE_full_data.py