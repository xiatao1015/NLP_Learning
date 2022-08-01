import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


class Word2Vec(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.W = nn.Linear(voc_size, embedding_size, bias=False)
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, x):
        hidden_layer = self.W(x)
        output_layer = self.WT(hidden_layer)
        return output_layer


def random_batch(skip_grams, batch_size, voc_size):
    random_inputs = []
    random_labels = []
    random_indexes = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_indexes:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])
        random_labels.append(skip_grams[i][1])

    return np.array(random_inputs), np.array(random_labels)


if __name__ == '__main__':
    batch_size = 2
    embedding_size = 2

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    sentences = ["apple banana fruit",
                 "banana orange fruit",
                 "orange banana fruit",
                 "dog cat animal",
                 "cat monkey animal",
                 "monkey dog animal"]

    word_token = " ".join(sentences).split()
    # ['apple', 'banana', 'fruit', 'banana', 'orange', 'fruit', 'orange', 'banana', 'fruit', 'dog', 'cat', 'animal',
    # 'cat', 'monkey', 'animal', 'monkey', 'dog', 'animal']
    word_list = list(set(word_token))
    word2idx = {word: index for index, word in enumerate(word_list)}
    # {'monkey': 0, 'animal': 1, 'fruit': 2, 'cat': 3, 'apple': 4, 'dog': 5, 'banana': 6, 'orange': 7}
    voc_size = len(word2idx)

    skip_gram = []
    for i in range(1, len(word_token) - 1):
        target = word2idx[word_token[i]]
        context = [word2idx[word_token[i - 1]], word2idx[word_token[i + 1]]]
        for word in context:
            skip_gram.append([target, word])

    model = Word2Vec(voc_size, embedding_size)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train
    for epoch in range(5000):
        input_batch, target_batch = random_batch(skip_gram, batch_size, voc_size)
        # print(input_batch, target_batch)
        input_batch = torch.Tensor(input_batch)
        input_batch = input_batch.to(device)
        target_batch = torch.Tensor(target_batch)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch.long())
        if (epoch + 1) % 1e3 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            print(output)

        loss.backward()
        optimizer.step()

        for i, label in enumerate(word_list):
            W, WT = model.parameters()
            x, y = W[0][i].item(), W[1][i].item()
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.show()