cd src/cuda/
/usr/local/cuda/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch='sm_61'
cd ../../
python build.py
cd ../../