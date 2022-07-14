import math
from argparse import ArgumentParser
import plotly.express as px
import torch
import wandb
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix
import seaborn as sns


class SConv1d(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size,
                 stride=1, pad=0,
                 drop=None, bn=True, activ=lambda: nn.PReLU()):
        super(SConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_filters, in_filters,
                                   kernel_size=kernel_size, groups=in_filters,
                                   stride=stride, padding=pad)
        self.pointwise = nn.Conv1d(in_filters, out_filters,
                                   kernel_size=1)
        layers = []
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(out_filters))
        if drop is not None:
            assert 0.0 < drop < 1.0
            layers.append(nn.Dropout(p=drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.layers(x)
        return x


class AttentionEEG(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("IMClassifier")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--in_channels", type=int, default=27)
        parser.add_argument("--n_classes", type=int, default=3)
        parser.add_argument("--n_persons", type=int, default=109)
        return parent_parser

    def __init__(self, in_channels: int,
                 n_classes: int,
                 n_persons: int,
                 drop=0.5, **kwargs):
        super().__init__()
        self.lr = kwargs['lr']
        self.n_persons = n_persons
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.hidden_channels = 32
        self.temporal_convs = 64

        # Raw
        self.r_sconv1d_1 = SConv1d(in_channels, in_channels, 8, 2, 3, bn=True, drop=drop)
        self.r_sconv1d_2 = SConv1d(in_channels, in_channels, 3, 1, 1, bn=True, drop=drop)
        self.r_sconv1d_3 = SConv1d(in_channels, in_channels, 8, 2, 3, bn=True, drop=drop)
        self.r_sconv1d_4 = SConv1d(in_channels, in_channels, 3, 1, 1, bn=True, drop=drop)
        # self.r_sconv1d_5 = SConv1d(in_channels, in_channels, 8, 2, 3, bn=True, drop=drop)

        hidden_aspp = 4
        self.r_aspp_1 = nn.Conv2d(1, hidden_aspp, kernel_size=1)
        self.r_aspp_2 = nn.Conv2d(1, hidden_aspp, kernel_size=3, dilation=(4, 8), padding=(4, 8))
        self.r_aspp_3 = nn.Conv2d(1, hidden_aspp, kernel_size=3, dilation=(8, 16), padding=(8, 16))
        self.r_aspp_4 = nn.Conv2d(1, hidden_aspp, kernel_size=3, dilation=(12, 24), padding=(12, 24))

        # concat -> (-1, 16, 27, 32)
        self.r_concat_conv = nn.Conv2d(hidden_aspp * 4, 1, kernel_size=1)

        # concat -> (-1, 1, 27, 32)
        self.r_bn = nn.BatchNorm2d(1)

        # IM
        # flatten -> (-1, 27 * 32)
        self.im_lin1 = nn.Linear(self.in_channels * self.temporal_convs, 64)
        self.im_drop1 = nn.Dropout()
        self.im_lin_class = nn.Linear(64, n_classes)

        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=self.n_classes)

        # Activations
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

    def forward(self, raw):
        # Raw
        raw_out = self.r_sconv1d_1(raw)
        raw_out = self.r_sconv1d_2(raw_out)
        raw_out = self.r_sconv1d_3(raw_out)
        raw_out = self.r_sconv1d_4(raw_out)
        # raw_out -> (-1, 27, 64)

        # raw_out = self.r_sconv1d_5(raw_out)
        # raw_out -> (-1, 27, 32)

        raw_attention = raw_out @ raw_out.permute(0, 2, 1)
        raw_attention /= self.in_channels  # math.sqrt(self.in_channels)
        softmaxes = torch.softmax(raw_attention, dim=2)

        # softmaxes -> (-1, 27, 27)

        raw_out = softmaxes @ raw_out
        # out -> (-1, 27, 32)

        raw_out = raw_out.unsqueeze(1)
        # out -> (-1, 1, 27, 32)

        raw_aspp1 = self.r_aspp_1(raw_out)
        raw_aspp2 = self.r_aspp_2(raw_out)
        raw_aspp3 = self.r_aspp_3(raw_out)
        raw_aspp4 = self.r_aspp_4(raw_out)
        # raw_aspp -> (-1, 4, 27, 32)

        raw_out = torch.concat((raw_aspp1, raw_aspp2, raw_aspp3, raw_aspp4), dim=1)
        # raw_out -> (-1, 16, 27, 32)

        raw_out = self.r_concat_conv(raw_out)
        raw_out = self.r_bn(raw_out)
        # raw_out -> (-1, 1, 27, 32)

        # out -> (-1, 32 * 27)
        raw_out = raw_out.view(-1, self.temporal_convs * self.in_channels)
        raw_out = self.im_drop1(self.activation(self.im_lin1(raw_out)))
        raw_out = self.im_lin_class(raw_out)
        raw_out = self.final_activation(raw_out)

        return raw_out

    def training_step(self, batch):
        raw_data, fft_data, target_im, target_person = batch

        im_predicted = self.forward(raw_data)

        im_loss = self.loss_func(im_predicted, target_im)
        im_accuracy = self.accuracy(torch.argmax(im_predicted, dim=1), target_im)

        conf_matrix = self.confusion_matrix(torch.argmax(im_predicted, dim=1), target_im)
        conf_matrix = conf_matrix.cpu().detach().numpy()
        fig = px.imshow(conf_matrix, text_auto=True)
        if self.global_step % 50 == 0:
            self.logger.experiment.log({'Matrix/Confusion Matrix': fig})

        # class_names = ['Rest', '1', 'Legs', '2']

        self.log("Train Loss", im_loss)
        self.log("Train Accuracy", im_accuracy, prog_bar=True)

        return im_loss

    def loss_func(self, y_pred, y_true):
        loss = nn.CrossEntropyLoss()
        return loss(y_pred, y_true)

    def validation_step(self, batch, idx, dataloader_idx):
        raw_data, fft_raw, im_target, person_target = batch

        im_predicted = self.forward(raw_data)

        loss = self.loss_func(im_predicted, im_target)
        accuracy = self.accuracy(im_predicted, im_target)

        conf_matrix = self.confusion_matrix(torch.argmax(im_predicted, dim=1), im_target)
        conf_matrix = conf_matrix.cpu().detach().numpy()
        fig = px.imshow(conf_matrix, text_auto=True)

        if dataloader_idx == 0:
            self.log("Val Loss", loss)
            self.log("Val Accuracy", accuracy, prog_bar=True)
            if idx % 50 == 0:
                self.logger.experiment.log({'Matrix/Val Confusion Matrix': fig})

        if dataloader_idx == 1:
            self.log("Test Loss", loss)
            self.log("Test Accuracy", accuracy, prog_bar=True)
            if idx % 50 == 0:
                self.logger.experiment.log({'Matrix/Test Confusion Matrix': fig})

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, total_steps=1000)
        return {"optimizer": optim,
                "scheduler": scheduler}


if __name__ == '__main__':
    raw = torch.randn((10, 27, 256))
    fft = torch.randn((10, 27, 129))
    attention = AttentionEEG(27, 3, n_persons=109, lr=0.0003, drop=0.5)
    x1, x2 = attention.training_step((raw, fft, torch.ones(10).long(), 1))

    print(x1.shape)
