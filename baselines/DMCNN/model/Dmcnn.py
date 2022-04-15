import torch
import torch.nn as nn
from utils.global_variables import Global


class Dmcnn(nn.Module):
    def __init__(self, config):
        super(Dmcnn, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.pf_embedding = nn.Embedding(num_embeddings=config.getint("runtime", "sequence_length"),
                                         embedding_dim=config.getint("model", "pf_dim"))
        self.cnn = _CNN(config)
        self.pooling = _DynamicPooling(config)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        self.fc = nn.Linear(in_features=config.getint("model", "llf_num") * config.getint("runtime", "embedding_size") + 2 * config.getint("model", "hidden_size"),
                            out_features=config.getint("runtime", "num_class"),
                            bias=True)
        self.out = OutputLayer(config)
        print(self)

    def forward(self, data, **params):
        """
        :param data: 这一轮输入的数据
        :param params: 存放任何其它需要的信息
        """
        mode = params["mode"]
        tokens = data["tokens"]
        if mode != "test":
            labels = data["labels"]
        masks = data["masks"]
        pfs = data["pfs"]
        llfs = data["llfs"]

        llf = self.embedding(llfs).view(-1, self.config.getint("model",
                                                               "llf_num") * self.config.getint("runtime", "embedding_size"))
        prediction = torch.cat(
            (self.embedding(tokens), self.pf_embedding(pfs)), dim=-1)    # [B, L, E+P]
        prediction = self.cnn(prediction)   # [B, H, L]
        prediction = self.pooling(prediction, masks)    # [B, 2*H]
        prediction = self.dropout(prediction)
        prediction = torch.cat((prediction, llf), dim=-1)   # [B, l*E+2*H]
        prediction = self.fc(prediction)    # [B, N]

        if mode != "test":
            loss = self.out(prediction, labels)
        prediction = torch.argmax(prediction, dim=1)

        return {"loss": loss,
                "prediction": prediction,
                "labels": labels} if mode != "test" else {
                    "prediction": prediction
        }


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        if Global.word2vec_mat is None:
            weight = None
            self.vocab_size = config.getint("runtime", "vocab_size")
            self.embedding_size = config.getint("runtime", "embedding_size")
        else:
            weight = torch.from_numpy(Global.word2vec_mat).float()
            self.vocab_size, self.embedding_size = weight.size()
        self.embedding = nn.Embedding.from_pretrained(weight)

    def forward(self, input):
        return self.embedding(input)


class OutputLayer(nn.Module):
    def __init__(self, config):
        super(OutputLayer, self).__init__()
        self.num_class = config.getint("runtime", "num_class")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, prediction, labels):
        loss = self.criterion(prediction, labels)   # ([B, N], [B,])
        return loss


class _CNN(nn.Module):
    def __init__(self, config):
        super(_CNN, self).__init__()
        self.in_channels = config.getint(
            "runtime", "embedding_size") + config.getint("model", "pf_dim")
        self.out_channels = config.getint("model", "hidden_size")
        self.kernel_size = config.getint("model", "kernel_size")
        self.padding_size = (self.kernel_size - 1) >> 1
        self.cnn = nn.Conv1d(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             kernel_size=self.kernel_size,
                             stride=1,
                             padding=self.padding_size)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)    # [B, L, E+P] -> [B, E+P, L]
        prediction = self.cnn(inputs)       # [B, E+P, L] -> [B, H, L]
        prediction = self.activation(prediction)    # [B, H, L]
        return prediction


class _DynamicPooling(nn.Module):
    def __init__(self, config):
        super(_DynamicPooling, self).__init__()
        self.hidden_size = config.getint("model", "hidden_size")

    def forward(self, inputs, masks):
        inputs = torch.unsqueeze(inputs, dim=-1)  # [B, H, L] -> [B, H, L, 1]
        masks = torch.unsqueeze(masks, dim=1)    # [B, L, 3] -> [B, 1, L, 3]
        prediction = torch.max(masks + inputs, dim=2)[0]
        prediction -= 100
        prediction = prediction.view(-1, 2 * self.hidden_size)
        return prediction
