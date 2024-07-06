import torch


class CAMS():
    def grad_cam(self, activations, output, normalization='relu_min_max', avg_grads=False, norm_grads=False):
        def normalize(grads):
            l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
            return grads * torch.pow(l2_norm, -1)

        # Obtain gradients
        gradients = torch.autograd.grad(output, activations, grad_outputs=None, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]

        # Normalize gradients
        if norm_grads:
            gradients = normalize(gradients)

        # pool the gradients across the channels
        if avg_grads:
            gradients = torch.mean(gradients, dim=[2, 3])
            # gradients = torch.nn.functional.softmax(gradients)
            gradients = gradients.unsqueeze(-1).unsqueeze(-1)

        # weight activation maps
        if 'relu' in normalization:
            GCAM = torch.sum(torch.relu(gradients * activations), 1)
        else:
            GCAM = gradients * activations
            if 'abs' in normalization:
                GCAM = torch.abs(GCAM)
            GCAM = torch.sum(GCAM, 1)

        # GCAM = torch.mean(activations, 1)

        # # Normalize CAM
        # if 'sigm' in normalization:
        #     GCAM = torch.sigmoid(GCAM)
        # if 'min' in normalization:
        #     norm_value = torch.min(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        #     GCAM = GCAM - norm_value
        # if 'max' in normalization:
        #     norm_value = torch.max(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        #     GCAM = GCAM * norm_value.pow(-1)
        # if 'tanh' in normalization:
        #     GCAM = torch.tanh(GCAM)
        # if 'clamp' in normalization:
        #     GCAM = GCAM.clamp(max=1)

        return GCAM


    def cam(self, activations, normalization='relu_min_max'):
        CAM = torch.mean(activations, 1)

        # Normalize CAM
        if 'sigm' in normalization:
            CAM = torch.sigmoid(CAM)
        if 'min' in normalization:
            norm_value = torch.min(torch.max(CAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
            CAM = CAM - norm_value
        if 'max' in normalization:
            norm_value = torch.max(torch.max(CAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
            CAM = CAM * norm_value.pow(-1)
        if 'tanh' in normalization:
            CAM = torch.tanh(CAM)
        if 'clamp' in normalization:
            CAM = CAM.clamp(max=1)

        return CAM