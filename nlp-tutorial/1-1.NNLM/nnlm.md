1. 通过输入的所有句子构建词表
2. 通过set去重后生成word2idx， idx2word
3. NNLM通过滑动窗口，选择n_step作为窗口大小，即论文中所说前（n-1）个词，去预测第n个词
4. 矩阵C: 词嵌入后的向量矩阵[len(word2idx), dim]
5. input: word2idx(word for word in (n-1)个词)
6. input经过C，后进行拼接得到[1, dim*(n-1)]
7. 通过简单全连接进入hidden layer：通过tanh激活函数(tanh = tanh(Hx+d))
8. output：又通过一个全连接层输出[len(word2idx), 1]后经过softmax得到概率