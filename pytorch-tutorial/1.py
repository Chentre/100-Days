# import torch
# if torch.cuda.is_available():
#     num_gpus = torch.cuda.device_count()
#     for i in range(num_gpus):
#         print(f"GPU {i} 型号: {torch.cuda.get_device_name(i)}")
# else:
#     print("未检测到可用的GPU")

# import torch
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))  # 检查第一块GPU的名称
# print(torch.cuda.get_device_name(1))  # 检查第二块GPU的名称，如果存在的话

import torch
if torch.cuda.device_count() > 0:
    print(torch.cuda.get_device_name(0))
    # 只有在有多个GPU时才尝试访问其他设备ID
    if torch.cuda.device_count() > 1:
        print(torch.cuda.get_device_name(1))
else:
    print("没有检测到GPU")