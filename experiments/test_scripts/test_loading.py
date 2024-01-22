from utils.loader import get_datasets

train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
train_mean = train1.dataset.data.float().mean()/255
train_std = train1.dataset.data.float().std()/255

print(train_mean)
print(train_std)