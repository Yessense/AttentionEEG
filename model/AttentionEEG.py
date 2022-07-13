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
        self.hidden_channels = 64
        self.temporal_convs = 128

        # Raw
        self.r_sconv1d_1 = SConv1d(in_channels, in_channels, 8, 2, 3, bn=True, drop=drop)
        self.r_sconv1d_2 = SConv1d(in_channels, in_channels, 3, 1, 1, bn=True, drop=drop)
        self.r_sconv1d_3 = SConv1d(in_channels, in_channels, 8, 2, 3, bn=True, drop=drop)
        self.r_sconv1d_4 = SConv1d(in_channels, in_channels, 3, 1, 1, bn=True, drop=drop)

        # FFT
        self.f_sconv1d_1 = SConv1d(in_channels, in_channels, 3, 1, 1, bn=True, drop=drop)
        self.f_sconv1d_2 = SConv1d(in_channels, in_channels, 8, 2, 3, bn=True, drop=drop)
        self.f_sconv1d_3 = SConv1d(in_channels, in_channels, 3, 1, 1, bn=True, drop=drop)
        self.f_sconv1d_4 = SConv1d(in_channels, in_channels, 8, 2, 3, bn=True, drop=drop)

        # Activations
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

        # temporal convs
        self.t_conv = nn.Conv1d(in_channels, self.temporal_convs, kernel_size=self.hidden_channels)
        self.t_bn = nn.BatchNorm1d(self.temporal_convs)

        # temporal convs
        self.fft_t_conv = nn.Conv1d(in_channels, self.temporal_convs, kernel_size=32)
        self.fft_t_bn = nn.BatchNorm1d(self.temporal_convs)

        # Person
        self.p_lin1 = nn.Linear(self.temporal_convs, 128)
        self.p_drop1 = nn.Dropout()
        self.p_lin_class = nn.Linear(128, self.n_persons)

        # IM
        self.im_lin1 = nn.Linear(self.temporal_convs, 64)
        self.im_drop1 = nn.Dropout()
        self.im_lin_class = nn.Linear(64, n_classes)

        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=self.n_classes)

    def forward(self, raw, fft):
        # Raw
        raw_out = self.r_sconv1d_1(raw)
        raw_out = self.r_sconv1d_2(raw_out)
        raw_out = self.r_sconv1d_3(raw_out)
        raw_out = self.r_sconv1d_4(raw_out)
        # raw_out -> (-1, 27, 64)

        # fft
        fft_out = self.f_sconv1d_1(fft)
        fft_out = self.f_sconv1d_2(fft_out)
        fft_out = self.f_sconv1d_3(fft_out)
        fft_out = self.f_sconv1d_4(fft_out)
        # fft_out -> (-1, 27, 32)

        combined = torch.cat((raw_out, fft_out), dim=2)
        raw_attention = combined @ combined.permute(0, 2, 1)
        raw_attention /= self.in_channels  # math.sqrt(self.in_channels)
        softmaxes = torch.softmax(raw_attention, dim=2)

        # softmaxes -> (-1, 27, 27)

        raw_out = softmaxes @ raw_out
        # out -> (-1, 27, 64)
        raw_out = self.t_conv(raw_out)
        # out -> (-1, 128, 1)
        raw_out = raw_out.view(-1, self.temporal_convs)
        # out -> (-1, 128)
        raw_out = self.t_bn(raw_out)

        fft_attention = fft_out @ fft_out.permute(0, 2, 1)
        fft_attention /= self.in_channels  # math.sqrt(self.in_channels)
        fft_softmax = torch.softmax(fft_attention, dim=2)
        # softmaxes -> (-1, 27, 27)

        fft_out = fft_softmax @ fft_out
        # out -> (-1, 27, 32)
        fft_out = self.fft_t_conv(fft_out)
        # out -> (-1, 128, 1)
        fft_out = fft_out.view(-1, self.temporal_convs)
        # out -> (-1, 128)
        fft_out = self.fft_t_bn(fft_out)

        person = self.p_drop1(self.activation(self.p_lin1(fft_out)))
        person = self.p_lin_class(person)

        raw_out = self.im_drop1(self.activation(self.im_lin1(raw_out)))
        raw_out = self.im_lin_class(raw_out)

        raw_out = self.final_activation(raw_out)
        person = self.final_activation(person)

        return raw_out, person

    def training_step(self, batch):
        raw_data, fft_data, target_im, target_person = batch

        im_predicted, person_predicted = self.forward(raw_data, fft_data)

        im_loss = self.loss_func(im_predicted, target_im)
        im_accuracy = self.accuracy(torch.argmax(im_predicted, dim=1), target_im)

        person_loss = self.loss_func(person_predicted, target_person)
        person_accuracy = self.accuracy(torch.argmax(person_predicted, dim=1), target_person)

        conf_matrix = self.confusion_matrix(torch.argmax(im_predicted, dim=1), target_im)
        conf_matrix = conf_matrix.cpu().detach().numpy()
        fig = px.imshow(conf_matrix, text_auto=True)
        if self.global_step % 50 == 0:
            self.logger.experiment.log({'Matrix/Confusion Matrix': fig})

        # class_names = ['Rest', '1', 'Legs', '2']

        self.log("Train Loss", im_loss)
        self.log("Person Loss", person_loss)
        self.log("Train Accuracy", im_accuracy, prog_bar=True)
        self.log("Person Accuracy", person_accuracy)

        return im_loss + 1 / 20 * person_loss

    def loss_func(self, y_pred, y_true):
        loss = nn.CrossEntropyLoss()
        return loss(y_pred, y_true)

    def validation_step(self, batch, idx, dataloader_idx):
        raw_data, fft_raw, im_target, person_target = batch

        im_predicted, person_predicted = self.forward(raw_data, fft_raw)

        loss = self.loss_func(im_predicted, im_target)
        accuracy = self.accuracy(im_predicted, im_target)

        conf_matrix = self.confusion_matrix(torch.argmax(im_predicted, dim=1), im_target)
        conf_matrix = conf_matrix.cpu().detach().numpy()
        fig = px.imshow(conf_matrix, text_auto=True)

        if dataloader_idx == 0:
            self.log("Val Loss", loss)
            self.log("Val Accuracy", accuracy, prog_bar=True)
            person_loss = self.loss_func(person_predicted, person_target)
            person_accuracy = self.accuracy(torch.argmax(person_predicted, dim=1), person_target)
            self.log("Person Loss", person_loss)
            self.log("Person Accuracy", person_accuracy)
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
    attention = AttentionEEG(27, 128, n_persons=109, lr=0.0003, drop=0.5)
    x1, x2 = attention.training_step((raw, fft, 1, 1))

    print(x1.shape)
