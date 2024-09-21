nvcc --version || echo "nvcc not found"
ldconfig -p | grep cuda || echo "CUDA libraries not found"
nvidia-smi || echo "nvidia-smi not found"
python3 --version

# Check for selective_scan_cuda
echo "Checking for selective_scan_cuda..."
find /usr/local/cuda/ -name "*selective_scan_cuda*" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "selective_scan_cuda found."
else
    echo "selective_scan_cuda not found."
fi

python3 /nethome/hhwang/hhwang/shiawaseda/M2_BPE_full_data.py