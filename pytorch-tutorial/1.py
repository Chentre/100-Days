import torch
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i} 型号: {torch.cuda.get_device_name(i)}")
else:
    print("未检测到可用的GPU")
