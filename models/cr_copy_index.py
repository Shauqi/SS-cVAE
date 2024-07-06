import torch

def copy_index(dy1,dy2,dy3,rand_index,gap=0.9,cuda=True,device='cuda:0'): 
	if gap > 0: 
		is_dy1_greater = torch.greater(dy1,dy2).type(torch.uint8)

		is_dy2_greater = torch.ones_like(is_dy1_greater) - is_dy1_greater

		#print(is_dy1_greater, is_dy2_greater)

		add_dy1_sub_dy2 = torch.mul(is_dy1_greater, gap/2)
		add_dy2_sub_dy1 = torch.mul(is_dy2_greater, gap/2)

		new_dy1 = torch.sub(torch.add(dy1, add_dy1_sub_dy2),add_dy2_sub_dy1)

		new_dy2 = torch.sub(torch.add(dy2,add_dy2_sub_dy1),add_dy1_sub_dy2)
	else: 
		new_dy1 = dy1
		new_dy2 = dy2

	size = dy1.size()
	vdy1 = new_dy1.view(size[0] * size[1])
	vdy3 = dy3.view(size[0] * size[1])
	vdy2 = new_dy2.view(size[0] * size[1])

	i = torch.arange(0,size[0], device=device)
	# if cuda : 
	# 	i = i.cuda()

	i2 = i.view((size[0],1))
	rand_index2 = rand_index.view((size[0],1))

	index = torch.cat((i2,rand_index2),dim=1)

	lin_index = index.select(1, 0) * size[1] + index.select(1, 1)

	vdy1.index_copy_(0, lin_index, vdy3.index_select(0, lin_index))
	vdy2.index_copy_(0, lin_index, vdy3.index_select(0, lin_index))

	return vdy1.view(size), vdy2.view(size)


if __name__ == '__main__':

	bs = 2 
	latent_dim = 3

	dy1 = torch.tensor([[0,8,2],[1,4,5]])

	dy2 = torch.tensor([[6,7,3],[2,9,1]])

	gap = 5

	print("dy1 = ", dy1)
	print("dy2 = ", dy2)


	rand_index = torch.randint(low=0, high=latent_dim , size=(bs,))

	rand_index = rand_index.type(torch.LongTensor)


	print("rand_index = ", rand_index)

	copy_dy1, new_dy2 = copy_index(dy1,dy2,rand_index, gap, False)

	print("copy_dy1 = ", copy_dy1)
	print("new_dy2 = ", new_dy2)