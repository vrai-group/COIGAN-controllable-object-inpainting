""" Full assembly of the parts to form the complete network """

from COIGAN.modules.unet.unet_parts import *


class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear=False, sigm_out=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigm_out = sigm_out
        factor = 2 if bilinear else 1

        # convolution
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # deconvolution
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # If requested add the sigmoid FDT to the output
        if sigm_out:
            self.outc = nn.Sequential(
                self.outc,
                nn.Sigmoid()
            )
        
        # Initialize the weights
        self.apply(self._init_weights)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
            
        #return self.outc(x)
        return {"out": self.outc(x)}
    
    
    def _init_weights(self, module):
        """
        Method that initialize all the the weights of conv2d and convTranspose2d
        layers with kaiming normal.
        """
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
