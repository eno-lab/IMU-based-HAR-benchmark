import torch
import torch.nn as nn

def get_config(dataset, lr_magnif=1):
    return {'n_hidden': 128,
            'n_filters': 64,
            'filter_size': 5,
            'drop_prob': 0.5,
            'cnn_depth': 4,
            'lstm_depth': 2,
            'learning_rate': 0.001 * lr_magnif,
            'regularization_rate': 0.01}

def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return torch_deep_conv_lstm(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'pytorch'


class torch_deep_conv_lstm(nn.Module):
    def __init__(self, 
              input_shape, 
              n_classes, 
              out_loss, 
              out_activ, 
              n_hidden=128, 
              n_filters=64, 
              filter_size=5, 
              drop_prob=0.5, 
              learning_rate=0.01, 
              cnn_depth=4,
              lstm_depth=2,
              metrics=['accuracy'],
              regularization_rate=0.01
              ):

        super(torch_deep_conv_lstm, self).__init__()

        self.input_shape = input_shape
        self.out_loss = out_loss
        self.learning_rate = learning_rate

        self.conv_layers = nn.ModuleList()
        for i in range(cnn_depth):
            _s = nn.Sequential(
                nn.Conv1d(input_shape[2] if i == 0 else n_filters, n_filters, filter_size),
                nn.ReLU()
                )
            self.conv_layers.append(_s)

        self.lstm_layers = nn.ModuleList()
        for i in range(lstm_depth):
            self.lstm_layers.append(nn.LSTM(n_filters if i == 0 else n_hidden, n_hidden, batch_first=True))

        self.dropout = nn.Dropout(drop_prob)
        self.linear_for_out = nn.Linear(n_hidden, n_classes)

        if out_activ == 'softmax':
            self.activation_for_out = nn.LogSoftmax(dim=1)
        else:
            raise NotImplementedError(f'Activation function {out_activ} is not implemented yet')


    def forward(self, x):
        # The input is [B, time-series, sensors] based on keras
        # torch needs channel first inputs. [B, sensors, time-series]
        x = torch.transpose(x, 2, 1) # swap to [B, Sensors, time-series]

        for l in self.conv_layers:
            x = l(x)

        x = torch.transpose(x, 2, 1) # [B, C, time] to [B, time, C]
        for l in self.lstm_layers:
            x, _ = l(x)

        x = self.dropout(x)
        x = x[:,-1,:] # extract values of final step 
        x = self.linear_for_out(x)
        x = self.activation_for_out(x)

        return x

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer


    def get_loss_function(self):
        if self.out_loss == 'categorical_crossentropy':
            ce_loss = nn.CrossEntropyLoss()
            def _CrossEntropy(x, y):
                y = torch.argmax(y, dim=1)
                return ce_loss(x, y)
            return _CrossEntropy
        else:
            raise NotImplementedError(f'Loss function {self.out_loss} is not implemented yet')


    def prepare_before_epoch(self, epoch):
        #epoch_tau = epoch+1
        #tau = max(1 - (epoch_tau - 1) / 50, 0.5)
        #for m in self.modules():
        #    if hasattr(m, '_update_tau'):
        #        m._update_tau(tau)
        pass

