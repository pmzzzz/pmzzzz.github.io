# 转置卷积


## 常用的图像增广方法

### 翻转和裁剪

```python
torchvision.transforms.RandomHorizontalFlip()
torchvision.transforms.RandomVerticalFlip()
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
```

### 改变颜色
```python
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
```
可以调节 ：图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
### 结合多种增广方法

在datasets里指定变换方法。
```python
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```
