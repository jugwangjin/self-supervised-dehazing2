# https://github.com/Reyhanehne/CVF-SID_PyTorch/blob/main/src/model/model.py

if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.utils import BoxBlur

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_downs=5

        class ResBlocks(nn.Module):
            def __init__(self, channels):
                super().__init__()
                conv1 = nn.Sequential(
                                    nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate'),
                                    nn.PReLU(channels)
                            )
                conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate')
                self.conv = nn.Sequential(conv1, conv2)

            def forward(self, x):
                return x + self.conv(x)

        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l1_encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, padding_mode='replicate'),
                                        nn.PReLU(32),
                                        )

        self.l2_encoder = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(64),
                                        )

        self.l3_encoder = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(128),
                                        )

        self.l4_encoder = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l5_encoder = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l4_skip = nn.Conv2d(256, 128, 1)
        self.l3_skip = nn.Conv2d(128, 64, 1)
        self.l2_skip = nn.Conv2d(64, 32, 1)
        self.l1_skip = nn.Conv2d(32, 16, 1)

        self.l5_up = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.l4_up = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.l3_up = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.l2_up = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.l1_up = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.l5_decoder = nn.Sequential(*[ResBlocks(256) for _ in range(2)])
        
        self.l4_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(384) for _ in range(2)]),
                                        nn.Conv2d(384, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l3_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(320) for _ in range(2)]),
                                        nn.Conv2d(320, 128, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(128),
                                        )
        
        self.l2_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(160) for _ in range(2)]),
                                        nn.Conv2d(160, 64, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(64),
                                        )
        
        self.l1_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(80) for _ in range(2)]),
                                        nn.Conv2d(80, 32, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(32),
                                        )

        self.out_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(32) for _ in range(2)]),
                                        nn.Conv2d(32, 6, 3, 1, 1, padding_mode='replicate'))
        

        self.conv_A = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(128, 3, 1, 1),
                                    nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()
        self.TBlur = BoxBlur(channels=3, kernel_size=5)

    def forward(self, x):
        x, pad = self.pad(x)

        enc_l1 = self.l1_encoder(x)
        

        enc_l2 = self.l2_encoder(self.down(enc_l1))
        

        enc_l3 = self.l3_encoder(self.down(enc_l2))
        

        enc_l4 = self.l4_encoder(self.down(enc_l3))
        

        enc_l5 = self.l5_encoder(self.down(enc_l4))


        enc_l5 = self.l5_decoder(enc_l5)

        dec_l4_inp = torch.cat((self.l4_skip(enc_l4), self.l5_up(enc_l5)), dim=1)
        
        dec_l4 = self.l4_decoder(dec_l4_inp)
        

        dec_l3_inp = torch.cat((self.l3_skip(enc_l3), self.l4_up(dec_l4)), dim=1)
        
        dec_l3 = self.l3_decoder(dec_l3_inp)


        dec_l2_inp = torch.cat((self.l2_skip(enc_l2), self.l3_up(dec_l3)), dim=1)
        
        dec_l2 = self.l2_decoder(dec_l2_inp)


        dec_l1_inp = torch.cat((self.l1_skip(enc_l1), self.l2_up(dec_l2)), dim=1)
        
        dec_l1 = self.l1_decoder(dec_l1_inp)

        out = self.l1_up(dec_l1)

        out = self.out_decoder(out)

        T, J = torch.split(out, 3, dim=1)
        J = J + x
        J = self.unpad(J, pad)
        T = self.unpad(T, pad)

        T = self.TBlur(T)
        T = self.sigmoid(T)
        A = self.conv_A(enc_l5)

        return T, A, J


    def pad(self, x):
        x_padx = ((2**self.num_downs) - (x.size(2)%(2**self.num_downs))) % (2**self.num_downs)
        x_pady = ((2**self.num_downs) - (x.size(3)%(2**self.num_downs))) % (2**self.num_downs)
        x = F.pad(x, [x_pady//2, (x_pady - x_pady//2), 
                    x_padx // 2, (x_padx - x_padx//2)], mode='replicate')

        return x, (x_padx, x_pady)

    def unpad(self, x, pad):

        x = x[:, :, pad[0] // 2 : x.size(2) - (pad[0] - pad[0] // 2), 
                pad[1] // 2 : x.size(3) - (pad[1] - pad[1] // 2)]
        return x     






class UNetBL(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_downs=5
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        class ResBlocks(nn.Module):
            def __init__(self, channels):
                super().__init__()
                conv1 = nn.Sequential(
                                    nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate'),
                                    nn.PReLU(channels)
                            )
                conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate')
                self.conv = nn.Sequential(conv1, conv2)

            def forward(self, x):
                return x + self.conv(x)

        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l1_encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, padding_mode='replicate'),
                                        nn.PReLU(32),
                                        )

        self.l2_encoder = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(64),
                                        )

        self.l3_encoder = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(128),
                                        )

        self.l4_encoder = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l5_encoder = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l4_skip = nn.Conv2d(256, 128, 1)
        self.l3_skip = nn.Conv2d(128, 64, 1)
        self.l2_skip = nn.Conv2d(64, 32, 1)
        self.l1_skip = nn.Conv2d(32, 16, 1)

        self.l5_up = nn.Sequential(self.up, nn.Conv2d(256, 256, 3, 1, 1))
        self.l4_up = nn.Sequential(self.up, nn.Conv2d(256, 256, 3, 1, 1))
        self.l3_up = nn.Sequential(self.up, nn.Conv2d(128, 128, 3, 1, 1))
        self.l2_up = nn.Sequential(self.up, nn.Conv2d(64, 64, 3, 1, 1))
        self.l1_up = nn.Sequential(self.up, nn.Conv2d(32, 32, 3, 1, 1))

        self.l5_decoder = nn.Sequential(*[ResBlocks(256) for _ in range(2)])
        
        self.l4_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(384) for _ in range(2)]),
                                        nn.Conv2d(384, 256, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(256),
                                        )

        self.l3_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(320) for _ in range(2)]),
                                        nn.Conv2d(320, 128, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(128),
                                        )
        
        self.l2_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(160) for _ in range(2)]),
                                        nn.Conv2d(160, 64, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(64),
                                        )
        
        self.l1_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(80) for _ in range(2)]),
                                        nn.Conv2d(80, 32, 3, 1, 1, padding_mode='replicate'),
                                        nn.PReLU(32),
                                        )

        self.out_decoder = nn.Sequential(nn.Sequential(*[ResBlocks(32) for _ in range(2)]),
                                        nn.Conv2d(32, 6, 3, 1, 1, padding_mode='replicate'))
        

        self.conv_A = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(128, 3, 1, 1),
                                    nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()
        self.TBlur = BoxBlur(channels=3, kernel_size=5)

    def forward(self, x):
        x, pad = self.pad(x)

        enc_l1 = self.l1_encoder(x)
        

        enc_l2 = self.l2_encoder(self.down(enc_l1))
        

        enc_l3 = self.l3_encoder(self.down(enc_l2))
        

        enc_l4 = self.l4_encoder(self.down(enc_l3))
        

        enc_l5 = self.l5_encoder(self.down(enc_l4))


        enc_l5 = self.l5_decoder(enc_l5)

        dec_l4_inp = torch.cat((self.l4_skip(enc_l4), self.l5_up(enc_l5)), dim=1)
        
        dec_l4 = self.l4_decoder(dec_l4_inp)
        

        dec_l3_inp = torch.cat((self.l3_skip(enc_l3), self.l4_up(dec_l4)), dim=1)
        
        dec_l3 = self.l3_decoder(dec_l3_inp)


        dec_l2_inp = torch.cat((self.l2_skip(enc_l2), self.l3_up(dec_l3)), dim=1)
        
        dec_l2 = self.l2_decoder(dec_l2_inp)


        dec_l1_inp = torch.cat((self.l1_skip(enc_l1), self.l2_up(dec_l2)), dim=1)
        
        dec_l1 = self.l1_decoder(dec_l1_inp)

        out = self.l1_up(dec_l1)

        out = self.out_decoder(out)

        T, J = torch.split(out, 3, dim=1)
        J = J + x
        J = self.unpad(J, pad)
        T = self.unpad(T, pad)

        T = self.TBlur(T)
        T = self.sigmoid(T)
        A = self.conv_A(enc_l5)

        return T, A, J


    def pad(self, x):
        x_padx = ((2**self.num_downs) - (x.size(2)%(2**self.num_downs))) % (2**self.num_downs)
        x_pady = ((2**self.num_downs) - (x.size(3)%(2**self.num_downs))) % (2**self.num_downs)
        x = F.pad(x, [x_pady//2, (x_pady - x_pady//2), 
                    x_padx // 2, (x_padx - x_padx//2)], mode='replicate')

        return x, (x_padx, x_pady)

    def unpad(self, x, pad):

        x = x[:, :, pad[0] // 2 : x.size(2) - (pad[0] - pad[0] // 2), 
                pad[1] // 2 : x.size(3) - (pad[1] - pad[1] // 2)]
        return x     





class ResNet_NoNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_downs = 3
        self.downscale = 2**self.num_downs

        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, padding_mode='replicate'), )
        self.conv2_1 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, padding_mode='replicate'), nn.LeakyReLU(),)
        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='replicate'), nn.LeakyReLU(),)   
        self.conv2_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='replicate'),)
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, padding_mode='replicate'), nn.LeakyReLU(),)
        self.conv3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),)

        self.conv3_3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, padding_mode='replicate'), nn.LeakyReLU(),)
        self.conv3_4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, padding_mode='replicate'),)

        class ResBlocks(nn.Module):
            def __init__(self, channels):
                super().__init__()
                conv1 = nn.Sequential(
                                    nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate'),
                                    nn.LeakyReLU()
                            )
                conv2 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='replicate')
                self.conv = nn.Sequential(conv1, conv2)

            def forward(self, x):
                return x + self.conv(x)

        resblocks = [ResBlocks(256) for _ in range(16)]
        self.resblocks = nn.Sequential(*resblocks)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv4_1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, padding_mode='replicate'))
        self.conv4_2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, padding_mode='replicate'))
        self.conv5_1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, padding_mode='replicate'),)
        self.relu5_1 = nn.LeakyReLU()
        self.conv5_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='replicate'),nn.LeakyReLU())
        self.conv5_3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, padding_mode='replicate'),nn.LeakyReLU())
        self.conv6_1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, padding_mode='replicate'),)
        self.relu6_1 = nn.LeakyReLU()
        self.conv6_2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'), nn.LeakyReLU())
        self.conv6_3 = nn.Sequential(nn.Conv2d(32, 6, 3, 1, 1, padding_mode='replicate'))

        self.sigmoid = nn.Sigmoid() # for T and A

        self.conv_A = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Conv2d(128, 3, 1, 1),
                                    nn.Sigmoid())

        self.TBlur = BoxBlur(channels=3, kernel_size=5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform(m.weight, gain=0.1)
               nn.init.constant(m.bias, 0)

    def pad(self, x):
        x_padx = ((self.downscale) - (x.size(2)%(self.downscale))) % (self.downscale)
        x_pady = ((self.downscale) - (x.size(3)%(self.downscale))) % (self.downscale)
        x = F.pad(x, [x_pady//2, (x_pady - x_pady//2), 
                    x_padx // 2, (x_padx - x_padx//2)], mode='replicate')

        return x, (x_padx, x_pady)

    def unpad(self, x, pad):

        x = x[:, :, pad[0] // 2 : x.size(2) - (pad[0] - pad[0] // 2), 
                pad[1] // 2 : x.size(3) - (pad[1] - pad[1] // 2)]
        return x             

    def forward(self, x):
        x_, pad = self.pad(x)

        feat1_1 = self.conv1_1(x_)
        feat2_1 = self.conv2_1(feat1_1)
        feat2_2 = self.conv2_2(feat2_1)
        feat2_3 = self.conv2_3(feat2_2)
        feat3_1 = self.conv3_1(feat2_3)
        feat3_2 = self.conv3_2(feat3_1)
        feat3_3 = self.conv3_3(feat3_2)
        feat3_4 = self.conv3_4(feat3_3)

        resblocks_out = self.resblocks(feat3_4) + feat3_4 # long range residual term
        feat4_1 = self.conv4_1(resblocks_out)
        feat4_1 = feat4_1 + feat3_4
        feat4_1 = self.upsample(feat4_1)

        feat4_2 = self.conv4_2(feat4_1)
        feat4_2 = feat4_2 + feat3_2
        feat4_2 = self.upsample(feat4_2)

        feat5_1 = self.conv5_1(feat4_2)
        feat5_1 = feat5_1 + feat2_3
        feat5_1 = self.relu5_1(feat5_1)
        feat5_2 = self.conv5_2(feat5_1)
        feat5_3 = self.conv5_3(feat5_2)
        feat5_3 = self.upsample(feat5_3)
        feat6_1 = self.conv6_1(feat5_3)
        feat6_1 = feat6_1 + feat1_1
        feat6_1 = self.relu6_1(feat6_1)
        feat6_2 = self.conv6_2(feat6_1)
        feat6_3 = self.conv6_3(feat6_2)

        out = self.unpad(feat6_3, pad)

        T, J = torch.split(out, 3, dim=1)

        T = self.TBlur(T)
        T = self.sigmoid(T)
        # J = J + x
        J = self.sigmoid(J)
        A = self.conv_A(resblocks_out)

        return T, A, J