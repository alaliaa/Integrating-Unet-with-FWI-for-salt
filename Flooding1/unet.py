from collections import OrderedDict

import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32,skip=False):
        super(UNet, self).__init__()
	
        self.skip = skip
        self.out_channel = out_channels
        features = init_features
        self.in_channels= in_channels 
        # ENCODER
        self.encoder1 = UNet._block(in_channels, features)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)


	# BOTTLENECK
        self.bottleneck = UNet._block(features * 8, features * 16)
        # self.bottleneck = UNet._block(features * 4, features * 8, name="bottleneck")
        # self.bottleneck = UNet._block(features * 2, features * 4, name="bottleneck")

	# DECODER
        self.upconv4 = nn.ConvTranspose1d(
           features * 16, features * 8, kernel_size=2, stride=2 ,output_padding=1
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2    # AA/ I added output_padding
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features)

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=7,padding='same'
        )

    # For input skip 
        self.convx  = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=7,padding='same'
        )
        
    # Skip connections 
        self.conv_skip1 = nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=3, padding=1)
        )
        self.conv_skip2 = nn.Sequential(
            nn.Conv1d(features, features * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features * 2),
        )
        self.conv_skip3 = nn.Sequential(
            nn.Conv1d(features * 2, features * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features * 4),
        )
        self.conv_skip4 = nn.Sequential(
            nn.Conv1d(features * 4, features * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features * 8),
        )
        self.conv_skip_bootleneck = nn.Sequential(
            nn.Conv1d(features * 8, features * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features * 16),
        )
        self.up_conv_skip4 = nn.Sequential(
            nn.Conv1d(features * 16, features * 8,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features * 8),
        )
        self.up_conv_skip3 = nn.Sequential(
            nn.Conv1d(features * 8, features * 4,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features * 4),
        )
        self.up_conv_skip2 = nn.Sequential(
            nn.Conv1d(features * 4, features * 2,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features * 2),
        )
        self.up_conv_skip1 = nn.Sequential(
            nn.Conv1d(features * 2, features  ,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features  ),
        )

        # self.Dropout = nn.Dropout(0.7) 
        self.Relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):

        skip = self.skip 
        if skip: # U-net with residual blocks 
            enc1 = self.encoder1(x) + self.conv_skip1(x)
            pl1 = self.pool1(enc1)
            enc2 = self.encoder2(pl1) + self.conv_skip2(pl1)
            pl2 = self.pool2(enc2) 
            enc3 = self.encoder3(pl2) + self.conv_skip3(pl2)
            pl3 = self.pool3(enc3)
            enc4 = self.encoder4(pl3) + self.conv_skip4(pl3)
            pl4 = self.pool4(enc4)
            bottleneck = self.bottleneck(pl4) +self.conv_skip_bootleneck(pl4)

            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4) + self.up_conv_skip4(dec4) 
            dec3 = self.upconv3(dec4) 
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3) + self.up_conv_skip3(dec3)  
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2) + self.up_conv_skip2(dec2)  
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1) + self.up_conv_skip1(dec1)


            out = self.conv(dec1) + x
            # out = self.conv(dec1) 

            if self.out_channel ==2 :
                out[:,0,:] = self.sigmoid(out[:,0,:])
                out[:,1,:] = self.sigmoid(out[:,1,:])
            else: 
                out = self.sigmoid(out)

        else:
            enc1 = self.encoder1(x) 
            pl1 = self.pool1(enc1)
            enc2 = self.encoder2(pl1)
            pl2 = self.pool2(enc2) 
            enc3 = self.encoder3(pl2) 
            pl3 = self.pool3(enc3)
            enc4 = self.encoder4(pl3)
            pl4 = self.pool4(enc4)
            bottleneck = self.bottleneck(pl4) 

            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4) 
            dec3 = self.upconv3(dec4) 
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2) 
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1) 

            # print(f"dec1 shape is {dec1.shape}")
            out = self.conv(dec1) 

            if self.out_channel ==2 :
                out[:,0,:] = self.sigmoid(out[:,0,:])
                out[:,1,:] = self.sigmoid(out[:,1,:])
            elif self.in_channels ==2 :
                out = self.tanh(out+self.convx(x))
            else:  
                # out = self.tanh(out+x) 
                out = self.sigmoid(out) 

        # print(out.shape)
        return out



    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True),
                        nn.BatchNorm1d(num_features=features),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True),
                        nn.BatchNorm1d(num_features=features),
                        nn.ReLU(inplace=True)    
        )




    # @staticmethod
    # def _block(in_channels, features, name):
    #     return nn.Sequential(
    #         OrderedDict(
    #             [
    #                 (
    #                     name + "conv1",
    #                     nn.Conv1d(
    #                         in_channels=in_channels,
    #                         out_channels=features,
    #                         kernel_size=3,
    #                         padding=1,
    #                         bias=True,
    #                     ),
    #                 ),
    #                 (name + "norm1", nn.BatchNorm1d(num_features=features)),
    #                 (name + "relu1", nn.ReLU(inplace=True)),
    #                 (
    #                     name + "conv2",
    #                     nn.Conv1d(
    #                         in_channels=features,
    #                         out_channels=features,
    #                         kernel_size=3,
    #                         padding=1,
    #                         bias=True,
    #                     ),
    #                 ),
    #                 (name + "norm2", nn.BatchNorm1d(num_features=features)),
    #                 (name + "relu2", nn.ReLU(inplace=True)),
    #             ]
    #         )
    #     )
