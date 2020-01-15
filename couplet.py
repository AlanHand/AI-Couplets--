
from model import Model
'''
        训练模型从这里开始,训练数据在data目录下,训练完之后的模型在./data/models目录下
'''
m = Model(
        './data/dl-data/couplet/train/in.txt',
        './data/dl-data/couplet/train/out.txt',
        './data/dl-data/couplet/test/in.txt',
        './data/dl-data/couplet/test/out.txt',
        './data/dl-data/couplet/vocabs',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='./data/dl-data/models/tf-lib/output_couplet',
        restore_model=False)

# m.train(5000000)
m.train(10) # 测试迭代10次就训练完
