# with torch.no_grad():
#     latent_dim = 22
#     n = 10
#     digit_size = 64
#     n_channels = 3
#     figure = np.zeros((digit_size * n, digit_size * n, n_channels))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.zeros((1, latent_dim))
#             z_sample[0][0] = xi
#             z_sample[0][1] = yi
#             z_sample = torch.tensor(z_sample).type(torch.FloatTensor).to(device)
#             x_decoded = model.decode(z_sample).cpu().numpy()
#             digit = x_decoded[0].reshape(digit_size, digit_size, n_channels)
#             print(digit.shape)
#             figure[i * digit_size: (i + 1) * digit_size,
#             j * digit_size: (j + 1) * digit_size, :] = digit

#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     # plt.xticks(pixel_range, sample_range_x)
#     # plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure)
#     plt.savefig("/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/output/MM_cVAE/sweep.png")


# with torch.no_grad():
#     n = 10
#     latent_dim = 22
#     digit_size = 64
#     n_channels = 3
#     figure = np.zeros((digit_size * n, digit_size * n, n_channels))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.zeros((1, latent_dim))
#             z_sample[0][0] = xi
#             z_sample[0][1] = yi
#             z_sample = torch.tensor(z_sample).type(torch.FloatTensor).to(device)
#             x_decoded = model.decode(z_sample).cpu().numpy()
#             digit = x_decoded[0].reshape(digit_size, digit_size, n_channels)
#             figure[i * digit_size: (i + 1) * digit_size,
#             j * digit_size: (j + 1) * digit_size, :] = digit

#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     # plt.xticks(pixel_range, sample_range_x)
#     # plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure)
#     plt.savefig("/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/output/MM_cVAE/sweep_1.png")

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.zeros((1, latent_dim))
#             z_sample[0][-1] = xi
#             z_sample[0][-2] = yi
#             z_sample = torch.tensor(z_sample).type(torch.FloatTensor).to(device)
#             x_decoded = model.decode(z_sample).cpu().numpy()
#             digit = x_decoded[0].reshape(digit_size, digit_size, n_channels)
#             figure[i * digit_size: (i + 1) * digit_size,
#             j * digit_size: (j + 1) * digit_size, :] = digit

#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     # plt.xticks(pixel_range, sample_range_x)
#     # plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("s[0]")
#     plt.ylabel("s[1]")
#     plt.imshow(figure)
#     plt.savefig("/ocean/projects/asc130006p/shared/mahmudul/Uncertainty_Estimation/output/MM_cVAE/sweep_2.png")
