import torch
import GPUtil
print(GPUtil.getAvailable())

print(torch.cuda.is_available())

print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__CUDA Device Name:',torch.cuda.get_device_name(0))
print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

device = torch.device("cuda")
print("Device: ",device)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(device=0))
