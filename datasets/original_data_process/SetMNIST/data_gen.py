from torchvision.datasets import MNIST
from torchvision import transforms
from matplotlib import pyplot as plt
import random as rand

size = 10
t = 0.3
data_dir = '.'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size)])
# transform = transforms.Compose([])
mnist = MNIST(data_dir, train=True, transform=transform, download=True)

# # 随机显示一些图片
# plt.figure(figsize=(10, 10))
# for i in range(1, 101):
#     data = mnist[rand.randint(0, len(mnist))][0][0]
#     data[data>=t]=1
#     data[data<t]=0
#     plt.subplot(10, 10, i)
#     plt.imshow(data)
# plt.show()

# # 统计集合大小
# data_sizes = []
# for img in mnist:
#     if img[1] !=8:continue
#     data = img[0][0]
#     data[data>=t]=1
#     data[data<t]=0
#     data_sizes.append((data>0).sum())
# print(sum(data_sizes) / len(data_sizes))
# print(max(data_sizes))
# plt.hist(data_sizes, bins=225)
# plt.show()

def gen_real_setmnist(digit=1):
    with open(f"setmnist{digit}.txt", "w", encoding='utf-8') as f:
        for img in mnist:
            data, n = img
            if n != digit:
                continue
            data = data[0]
            data[data>=t]=1
            data[data<t]=0
            eles = data.nonzero().numpy().tolist()
            eles_str = ','.join([f"@[{' '.join(map(str,pos))}]" for pos in eles])
            f.write(eles_str + "\n")

gen_real_setmnist(1)
gen_real_setmnist(8)

