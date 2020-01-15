from model import Model

'''
        测试模型效果
'''

vocab_file = './data/dl-data/couplet/vocabs'
model_dir = './data/dl-data/models/tf-lib/output_couplet'

m = Model(
        None, None, None, None, vocab_file,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)

output = m.infer(' '.join('天王盖地虎'))
print('下联',output)
print('上联：%s；下联：%s' % ('天王盖地虎', output))