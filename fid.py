import torch
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models
import numpy as np
from scipy.linalg import sqrtm
from torchvision import datasets, transforms
import torch.nn.functional as F
def preprocess_image(imgs):
    res = []
    for img in imgs:
        # img = np.array(img).astype(np.float32)
        if img.ndim == 2:
            img = torch.cat([img, img, img], dim=2)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        res.append(img)
    imgs = torch.stack(res, 0)
    imgs = F.interpolate(imgs, size=(128, 128), mode='bilinear', align_corners=False)
    return imgs

def calculate_activation_statistics(images, model):
    model.eval()
    with torch.no_grad():
        features = model(images)
        features = features.reshape(features.shape[0], -1)
        mean = features.mean(dim=0)
        cov = torch.matmul(features.T, features) / features.shape[0] - torch.matmul(mean.unsqueeze(1), mean.unsqueeze(0))
    return mean.cpu().numpy(), cov.cpu().numpy()

def calculate_frechet_distance(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(cov1.dot(cov2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(cov1 + cov2 - 2*covmean)

def calculate_fid(images_real, images_fake, model):
    mu1, cov1 = calculate_activation_statistics(preprocess_image(images_real), model)
    mu2, cov2 = calculate_activation_statistics(preprocess_image(images_fake), model)
    fid = calculate_frechet_distance(mu1, cov1, mu2, cov2)
    return fid

def get_fid_tool():
    return models.inception_v3(pretrained=True, aux_logits=False)

if __name__ == '__main__':
    # Example usage
    real_images = torch.randn((500, 3, 32, 32))
    fake_images = torch.randn((500, 3, 32, 32))
    real_images = preprocess_image(real_images)
    model = models.inception_v3(pretrained=True, aux_logits=False)
    fid = calculate_fid(real_images, fake_images, model)
    print('FID: ', fid)

