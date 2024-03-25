# import math
# import numbers
# import torch
# from torch import nn
# from torch.nn import functional as F
# import matplotlib.pyplot as plt
# import numpy as np



# #https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
# class GaussianConv(nn.Module):
#     def __init__(self, kernel_size, sigma, channels=1, dim=2, dtype=torch.float32, device='cuda:0'):
#         super(GaussianConv, self).__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim

#         # The gaussian kernel is the product of the
#         # gaussian function of each dimension.
#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [
#                 torch.arange(size, dtype=torch.float32, device=device)
#                 for size in kernel_size
#             ]
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
#                       torch.exp(-((mgrid - mean) / std) ** 2 / 2)
            
#         # Make sure sum of values in gaussian kernel equals 1.
#         kernel = kernel / torch.sum(kernel)

#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer('weight', kernel)
#         self.groups = channels
#         self.padding = [ks//2 for ks in kernel_size]
#         self.kernel_size = [ks for ks in kernel_size]

#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(
#                 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
#             )
#         self.to(device)

#     def forward(self, input):
#         return self.conv(torch.unsqueeze(input.float(),1), weight=self.weight, padding=self.padding, groups=self.groups).view(input.shape)
    


# '''
#     On_Off_encoding
#     inspired by Center-surround receptive field
    
#     [1]	M. Van Wyk, H. Wässle, and W. R. Taylor, 
#         "Receptive field properties of ON- and OFF-ganglion cells in the mouse retina,"
#         Vis. Neurosci., vol. 26, no. 3, pp. 297-308, 2009.

# '''

# # input: (channel, siz)  numpy.darray
# # T: during
# def On_Off_encoding(input, T):   
#     input = torch.from_numpy(input)
#     batch_size = input.shape[0]
#     input = input.reshape((-1,28,28))

#     center_cells = GaussianConv(kernel_size=5, sigma=2, device='cpu')
#     surround_cells = GaussianConv(kernel_size=9, sigma=3, device='cpu')

#     img = center_cells(input) - surround_cells(input)
    
#     img = (img-img.median())/(img.std() + 1e-12)
#     img_final = torch.tanh(img)

#     # plt.imshow(img[0])
#     # plt.show()

#     # set near-zero responses to zero
#     on_cells = torch.where(img_final>0.05, img_final, torch.zeros_like(img_final)).reshape(batch_size, -1)
#     off_cells = torch.where(img_final<-0.05, img_final.abs(), torch.zeros_like(img_final)).reshape(batch_size, -1)

#     # plt.imshow(on_cells[0].reshape(28,28))
#     # plt.show()
#     # plt.imshow(off_cells[0].reshape(28,28))
#     # plt.show()

#     img_final = torch.stack([on_cells, off_cells]).permute(1,0,2)
    

#     # value for current
#     T_out = torch.stack([on_cells for t in range(T)])

#     return T_out.numpy()


#     # n_bins = 8
#     # T_out = torch.stack([(t%n_bins == (n_bins - torch.floor(on_cells * n_bins) - 1)) * torch.sign(on_cells) 
#     #                      for t in range(T)])
#     # for i in range(T):
#     #     plt.subplot(3,4,i+1)
#     #     plt.imshow(T_out[i,0,:].reshape(28,28))
#     # plt.show()

