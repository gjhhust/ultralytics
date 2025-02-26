import torch
import time
import signal
import sys

# 设置要占用的显卡和内存大小
device_ids = [2,3]
memory_size = 10 * 1024 * 1024 * 1024  # 20GB

# 用于存储分配的张量
allocated_tensors = {}

def allocate_memory(device_id, size):
    """在指定显卡上分配指定大小的内存"""
    tensor = torch.randn(size // 4, device=f'cuda:{device_id}')  # 每个float32元素占4字节
    return tensor

def release_memory():
    """释放所有已分配的内存"""
    for device_id, tensor in allocated_tensors.items():
        del tensor
        torch.cuda.empty_cache()
        print(f"已释放显卡 {device_id} 上的内存")

def signal_handler(sig, frame):
    """处理Ctrl+C中断信号"""
    print('你按下了 Ctrl+C!')
    release_memory()
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理程序
    signal.signal(signal.SIGINT, signal_handler)

    # 分配内存
    for device_id in device_ids:
        allocated_tensors[device_id] = allocate_memory(device_id, memory_size)
        print(f"已在显卡 {device_id} 上分配 {memory_size // (1024 ** 3)} GB 内存")

    # 保持脚本运行，直到手动停止
    print("脚本正在运行，按 Ctrl+C 停止并释放内存...")

    # 添加循环，不断进行计算以提高GPU利用率
    while True:
        for device_id in device_ids:
            # 在每个显卡上进行一些计算，例如矩阵乘法
            matrix = torch.randn(20480, 20480, device=f'cuda:{device_id}')
            result = torch.matmul(matrix, matrix)
            del matrix, result  # 删除不再需要的变量
            torch.cuda.empty_cache() # 清空CUDA缓存
        time.sleep(0.1)  # 减少循环间隔，提高GPU利用率