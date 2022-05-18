from torch import nn


class Conv1dBlock(nn.Module):
    """
    Conv 1-D block
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 padding):

        super(Conv1dBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.dconv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.PReLU(),
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, sample):
        conv1 = self.conv1(sample)
        norm1 = self.norm1(conv1)
        dconv = self.dconv(norm1)
        norm2 = self.norm2(dconv)
        conv2 = self.conv2(norm2)
        return conv2 + sample


class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, kernel_size):
        super(Encoder, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=kernel_size, padding=1, bias=False)
        self.prelu = nn.PReLU()

    def forward(self, sample):
        out = self.prelu(self.conv1d(sample))
        return out


class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, kernel_size):
        super(Decoder, self).__init__()

        self.convt = nn.ConvTranspose1d(in_channels=512, out_channels=80, kernel_size=kernel_size, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, data, mask):
        out = data * mask
        out = self.prelu(self.convt(out))
        return out


class AudioConvNet(nn.Module):
    """
    Model for denoising
    """
    def __init__(self, kernel_size=3, num_dilation=4, repeats=2):
        super(AudioConvNet, self).__init__()

        self.encoder = Encoder(kernel_size=kernel_size)
        self.decoder = Decoder(kernel_size=kernel_size)

        conv1x1 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        mask_conv1x1 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        blocks_repeats = []
        for r in range(repeats):
            blocks = []
            for x in range(num_dilation):
                dilation = 2 ** x
                padding = (kernel_size - 1) * dilation // 2
                blocks += [Conv1dBlock(in_channels=256,
                                       out_channels=512,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       dilation=dilation)]
            blocks_repeats += [nn.Sequential(*blocks)]
        conv_net = nn.Sequential(*blocks_repeats)

        self.network = nn.Sequential(nn.BatchNorm1d(512),
                                     conv1x1,
                                     conv_net,
                                     mask_conv1x1,
                                     nn.ReLU())

    def forward(self, x):

        data = self.encoder(x)
        mask = self.network(data)
        out = self.decoder(data, mask)

        return out
