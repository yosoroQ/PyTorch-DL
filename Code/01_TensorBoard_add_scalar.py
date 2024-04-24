# TensorBoard的使用
from torch.utils.tensorboard import SummaryWriter
 
writer = SummaryWriter("Code/logs")
for i in range(100):
    writer.add_scalar("y=2x",2*i,i) #标签y=2x，y轴2*i，x轴i

writer.close()

# python Code/01_TensorBoard_add_scalar.py

# # logdir为logs的目录路径，port端口号
# tensorboard --logdir=Code/logs --port=6007