import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2  as transforms


# データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True
)

print(f'dataset size: {len(ds_train)}')

# インデックスを指定してデータを取り出す
# 画像とクラス番号の組になっている
image, target = ds_train[0]

print(type(image))
print(target)

# 表示
plt.imshow(image, cmap='gray_r', )
plt.title(target)
plt.show()

# fig, ax = plt.subplots()
# ax.imshpw(image, cmap='gray_r)
# ax.set_titls(target)
# plt.show()

image = transforms.functional.to_image(image)
image = transforms.functional.to_image(image, scale=True)
print(type(image))
print(image.shape, image.dtype)
print(image.imn(), image.max())