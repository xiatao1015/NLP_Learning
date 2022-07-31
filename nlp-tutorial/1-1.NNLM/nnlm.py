import torch
import torch.nn as nn
import torch.optim as optim


class NNLM(nn.Module):
    """
    :param n_step: 输入词向量的个数
    :param n_class: 词表中的总次数。C的行数
    :param m: 向量维度
    :param n_hidden: 隐藏层节点数
    """

    def __init__(self, n_class, n_step, m, n_hidden):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(num_embeddings=n_class, embedding_dim=m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

        self.n_class = n_class
        self.n_step = n_step
        self.m = m
        self.n_hidden = n_hidden

    def forward(self, x):
        x = self.C(x)  # x: [batch_size, n_step, m]
        x = x.view(-1, self.n_step * self.m)  # x: [batch_size, n_step*m]
        tanh = torch.tanh(self.d + self.H(x))  # [batch_size, n_hidden]
        output = self.b + self.W(x) + self.U(tanh)  # [batch_size, n_class]
        return output


def make_batch(sentences, word2idx):
    input_batch = []
    target_batch = []

    for sentence in sentences:
        word = sentence.split()
        input = [word2idx[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word2idx[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


if __name__ == '__main__':
    sentences = ['i like dog',
                 'i dislike cat',
                 'i hate milk']
    n_step = 2
    m = 2
    n_hidden = 2

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    word_list = ' '.join(sentences).split()
    # word_list = ['i', 'like', 'dog', 'i', 'like', 'cat', 'i', 'hate', 'milk']
    word_list = list(set(word_list))
    # word_list = ['milk', 'like', 'i', 'hate', 'dog', 'cat']
    word2idx = {word: index for index, word in enumerate(word_list)}
    # word2idx = {'i': 0, 'milk': 1, 'like': 2, 'cat': 3, 'dog': 4, 'hate': 5}
    idx2word = {index: word for index, word in enumerate(word_list)}

    n_class = len(word2idx)

    nnlm = NNLM(n_class, n_step, m, n_hidden)
    nnlm = nnlm.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(nnlm.parameters(), lr=1e-3)

    input_batch, target_batch = make_batch(sentences, word2idx)
    input_batch = torch.LongTensor(input_batch)
    input_batch = input_batch.to(device)
    target_batch = torch.LongTensor(target_batch)
    target_batch = target_batch.to(device)

    # training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = nnlm(input_batch)

        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:' '%04d'%(epoch+1), 'cost=', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = nnlm(input_batch).data.max(1, keepdim=True)[1]

    print([sen.split()[:2] for sen in sentences], '->', [idx2word[n.item()] for n in predict.squeeze()])