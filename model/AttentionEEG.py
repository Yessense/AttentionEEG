import torch
from torch import nn
import pytorch_lightning as pl


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
    def __init__(self, raw_in, fft_in, out, drop=0.5):
        super().__init__()
        self.raw = nn.Sequential(
            SConv1d(raw_in, 32, 8, 2, 3, drop=drop),
            SConv1d(32, 32, 3, 1, 1, drop=drop),
            SConv1d(32, 64, 8, 4, 2, drop=drop),
            SConv1d(64, 64, 3, 1, 1, drop=drop),
            SConv1d(64, 128, 8, 4, 2, drop=drop),
            SConv1d(128, 128, 3, 1, 1, drop=drop),
            SConv1d(128, 256, 8, 4, 2),
            nn.Flatten(),
            nn.Dropout(drop), nn.Linear(512, 64), nn.PReLU(), nn.BatchNorm1d(64),
            nn.Dropout(drop), nn.Linear(64, 64), nn.PReLU(), nn.BatchNorm1d(64)
        )
        self.fft = nn.Sequential(
            SConv1d(fft_in, 32, 8, 2, 4, drop=drop),
            SConv1d(32, 32, 3, 1, 1, drop=drop),
            SConv1d(32, 64, 8, 2, 4, drop=drop),
            SConv1d(64, 64, 3, 1, 1, drop=drop),
            SConv1d(64, 128, 8, 4, 4, drop=drop),
            SConv1d(128, 128, 8, 4, 4, drop=drop),
            SConv1d(128, 256, 8, 2, 3),
            nn.Flatten(),
            nn.Dropout(drop), nn.Linear(512, 64), nn.PReLU(), nn.BatchNorm1d(64),
            nn.Dropout(drop), nn.Linear(64, 64), nn.PReLU(), nn.BatchNorm1d(64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, out))

    def forward(self, raw, fft):
        raw_out = self.raw(raw)
        fft_out = self.fft(fft)

        return raw_out, fft_out


if __name__ == '__main__':
    raw = torch.randn((2, 27, 256))
    fft = torch.randn((2, 27, 129))
    attention = AttentionEEG(27, 27, out=4, drop=0.5)
    x1, x2 = attention(raw, raw)

    print(x1.shape)
    print(x2.shape)
