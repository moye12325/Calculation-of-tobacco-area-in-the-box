import GPUtil

# 获取显卡信息列表
gpus = GPUtil.getGPUs()

# 遍历显卡信息列表
for gpu in gpus:
    # 打印显卡型号
    print("显卡型号：", gpu.name)
    # 打印其他显卡信息（可选）
    print("显卡ID：", gpu.id)
    print("显卡负载：", gpu.load * 100, "%")
    print("显存总大小：", gpu.memoryTotal, "MB")
    print("显存总大小：", gpu.memoryTotal/1024, "GB")
    print("显存已使用大小：", gpu.memoryUsed, "MB")
    print("显存剩余大小：", gpu.memoryFree, "MB")
    print("显卡温度：", gpu.temperature, "℃")
    print("————————————————————————")