import scipy.ndimage
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.nn import L1Loss

from torchvision import transforms


def inv_norm():
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]),
        transforms.ToPILImage()
    ])


def init_conv_weights(m, activations='relu'):

    gain = torch.nn.init.calculate_gain(activations)

    if type(m) == torch.nn.Conv2d  \
        or type(m) == torch.nn.ConvTranspose2d:

        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0.0)

# adapted from https://discuss.pytorch.org/t/gaussian-kernel-layer/37619
class GaussianLayer(torch.nn.Module):
    def __init__(self, sigma):

        super(GaussianLayer, self).__init__()

        self.sigma = sigma

        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(10),
            torch.nn.Conv2d(3, 3, 21, stride=1, padding=0, bias=None,
                            groups=3))

        self.weights_init()

    def forward(self, x):
        # print("==========================x in guss layer: ", x.shape)
        return self.seq(x)

    def weights_init(self):
        # keep it big, weights will tend to zero if sigma is small
        n = np.zeros((21, 21))
        n[10, 10] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=self.sigma)
        for _, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


class Smooth3DLayer(torch.nn.Module):
    def __init__(self, kernel_size=3, device='cuda:0'):
        super(Smooth3DLayer, self).__init__()

        # Ensure the kernel size is odd
        # kernel_size = torch.tensor(kernel_size, dtype=torch.float64)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = kernel_size

        # Create a 3D box filter kernel
        kernel = torch.full((kernel_size, kernel_size, kernel_size), 1.0 / (kernel_size ** 3), dtype=torch.float32)

        # Add batch and channel dimensions to the kernel
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        # Register the kernel as a parameter, to move it to the correct device
        self.register_buffer('kernel', kernel.to(device))

    def forward(self, x):
        # print('|||||||||||||||||||||||||||   in Smoother Before:', x.shape)
        x_tmp = x.view(-1, 1, 121, 145, 121)
        padding = self.kernel_size // 2
        res = F.conv3d(x_tmp, self.kernel, padding=padding)
        # print('|||||||||||||||||||||||||||   in Smoother After:', res.shape)
        return res.view(-1, 1, 1, 121, 145, 121)

# Usage example:
# smooth_layer = Smooth3DLayer(kernel_size=3)
# smoothed_tensor = smooth_layer(tensor)

def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def enable_grads(model):
    for p in model.parameters():
        p.requires_grad = True
    model.train()


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Adapted from https://github.com/jik0730/VAT-pytorch/blob/a7424f2ff386ceb39f80053c4103f9cd505ea07c/vat.py
def kl_divergence_with_logit(q_logit, p_logit):
    # print(q_logit, p_logit)
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    # print(qlogq.dtype, "   ", qlogp.dtype)
    return qlogq - qlogp


def train_adversarial_examples(x,
                               d,
                               a,
                               num_domains,
                               args,
                               model,
                               domain_adversary,
                               d_loss_fn,
                               d_loss_weight,
                               device,
                               epoch=None,
                               i=None,
                               wandb=None,
                               heldout=None):
    
    model.double()
    model_device = 'cuda:2'
    disable_grads(model)
    disable_grads(domain_adversary)
    adv_mask = torch.torch.bernoulli(args.adversarial_examples_ratio *
                                     torch.ones(x.size(0))).bool().to(device)
    if args.no_adversary_on_original:
        adv_mask = adv_mask & a
    # print('======================shape:' ,x.size(0))
    adv_x = x[adv_mask]
    d_orig = None
    if args.classify_adv_exp:
        d_orig = d.clone()
        d_orig[adv_mask] = d_orig[adv_mask] + num_domains

    if args.save_adversarial_examples and (not args.adv_img_saved_this_epoch):
        Path(f'results/{args.save_dir}/adv_examp/{heldout}/{epoch}').mkdir(
            parents=True, exist_ok=True)
        for j in range(adv_x.size(0)):
            img = adv_x[j].data.cpu()
            inv_norm()(img).save(
                f'results/{args.save_dir}/adv_examp/{heldout}/{epoch}/{i}-{j}-not.png'
            )
    # print("adv_v n elements", adv_x)
    if adv_x.nelement() > 0:
        torch.cuda.empty_cache()

        # print('======================shape:' ,adv_x.shape)
        # model.batch_size = adv_mask[0]
        old_class_distr = model(adv_x.double().to(model_device)).detach().to(device)
        adv_x = torch.nn.Parameter(adv_x)
        adv_x.requires_grad = True
        optim = torch.optim.Adam([adv_x],
                                 lr=args.adversarial_examples_lr,
                                 weight_decay=args.adversarial_examples_wd)
        for j in range(args.adversarial_train_steps):
            if not args.ablate_blur:
                # if(adv_x.shape[0] != len(x[adv_mask])):
                #     print(adv_x.shape[0], len(x[adv_mask]))
                #     adv_x = adv_x.view(adv_x.shape[1], adv_x.shape[2], adv_x.shape[3], adv_x.shape[4], adv_x.shape[5])
                # print("=-=-=-=-=-=-=-=-=-= adv_x: ", adv_x.shape)
                z_conv = model.conv_features(adv_x.double().to(model_device)).to(device)
                z = model.dense_features(z_conv.double().to(model_device)).to(device)
                # new_class_distr = F.log_softmax(model.classifier(z), dim=1)
                # print("=-=-=-=-=-=-=-=-=-= z_conv: ", z_conv.shape)
                # print("=-=-=-=-=-=-=-=-=-= z: ", z.shape)
                new_class_distr = model.classifier(z.double().to(model_device)).to(device)
                if args.dann_conv_layers:
                    dhat = domain_adversary(z_conv.double(), use_grad_reverse=False).to(device)
                else:
                    dhat = domain_adversary(z.double(), use_grad_reverse=False).to(device)
                optim.zero_grad()
                # print("--------------------inputs shape:", dhat.shape, d[adv_mask].shape, d_loss_weight)
                d_loss = d_loss_fn(dhat.double(), d[adv_mask].long()) * d_loss_weight
                # print("d_loss:", d_loss)
                kl_loss_ = args.adv_kl_weight * kl_divergence_with_logit(
                    old_class_distr, new_class_distr)
                kl_loss_ =  kl_loss_.double()
                # print("k_loss: ", d_loss.shape, kl_loss_.shape)
                ((d_loss + kl_loss_).double()).backward()

                if wandb is not None:
                    assert heldout is not None
                    wandb.log({
                        f"{heldout}_step": i,
                        f"{heldout}_adv_exp_d_loss": d_loss,
                        f"{heldout}_kl_loss": kl_loss_
                    })
                optim.step()
                # max and min of normed tensors in
                # PACS data set
                adv_x.data = adv_x.data.clamp(-2.1179, 2.64)
            if args.dataset == 'OpenBHB' and args.adv_smoothing_step > 0 and \
                j % args.adv_smoothing_step == 0:
                smooth_3d = Smooth3DLayer(args.adv_smoothing_strength)
                adv_x.data = smooth_3d(adv_x.data)
                # Reset gradients after modification
                optim.zero_grad()
            
                
            elif args.dataset != 'OpenBHB' and args.adv_blur_step > 0 and j % args.adv_blur_step == 0:
                blur = GaussianLayer(args.adv_blur_sigma).to(device)
                disable_grads(blur)
                adv_x.data = blur(adv_x.data)
                # just to make sure we don't get any grads
                # from blurring
                optim.zero_grad()
            # print("=-=-=-=-=-=-=-=-=-= adv_x =-=-=-=-: ", adv_x.shape)
        if args.dataset == 'OpenBHB' and args.adv_smoothing_step > 0 and args.smooth_at_last_step:
                smooth_3d = Smooth3DLayer(kernel_size=args.adv_smoothing_strength)
                adv_x.data = smooth_3d(adv_x.data)
                # Reset gradients after final smoothing
                optim.zero_grad()
        elif args.dataset != 'OpenBHB' and args.adv_blur_step > 0 and args.blur_at_last_step:
            blur = GaussianLayer(args.adv_blur_sigma).to(device)
            disable_grads(blur)
            adv_x.data = blur(adv_x.data)
            # just to make sure we don't get any grads
            # from blurring
            optim.zero_grad()

    if args.save_adversarial_examples and (not args.adv_img_saved_this_epoch):
        Path(f'results/{args.save_dir}/adv_examp/{heldout}/{epoch}').mkdir(
            parents=True, exist_ok=True)
        for j in range(adv_x.size(0)):
            img = adv_x[j].data.cpu()
            inv_norm()(img).save(
                f'results/{args.save_dir}/adv_examp/{heldout}/{epoch}/{i}-{j}-adv.png'
            )

    x[adv_mask] = adv_x.data

    enable_grads(model)
    enable_grads(domain_adversary)

    return x, d_orig


def scores_to_pixelval(scores, orig, a=-2.1179, b=2.64):
    # by default, a = min, b = max of PACS dataset
    # orig = orignal pixel vals
    return torch.sigmoid(scores) * (b - a) + a


def evaluate(model, dset, device, c_loss_fn, _save_dir):
    num_correct = 0
    num_total = 0
    total_loss = 0
    with torch.no_grad():
        for batch in dset:
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            model.eval()
            yhat = model(x)
            total_loss += c_loss_fn(yhat, y)
            num_correct += (yhat.argmax(dim=1) == y).sum().item()
            num_total += x.size(0)
        loss = total_loss / len(dset)
        acc = num_correct / num_total
    return acc, loss

def evaluate_regression(model, dset, device, c_loss_fn, _save_dir):
    num_correct = 0
    num_total = 0
    total_loss = 0
    MSE = 0
    MAE = 0
    i = 0
    MAE_func = L1Loss()
    with torch.no_grad():
        for batch in dset:
            i+=1
            x = batch[0].double().to(device)
            y = batch[1][:,0].double().to(device)
            # y = y.view(-1, 1)
            model.eval()
            yhat = model(x).to(device)
            yhat = yhat.view(-1)
            MSE += c_loss_fn(yhat, y)
            MAE += MAE_func(yhat, y)
            # print("MSE: ", MSE)
            # print("MAE: ", MAE)
            # total_loss += c_loss_fn(yhat, y)
            # num_correct += (yhat.argmax(dim=1) == y).sum().item()
            # num_total += x.size(0)
        # loss = total_loss / len(dset)
        # acc = num_correct / num_total
    MSE = MSE/i
    MAE = MAE/i
    return MAE, MSE

    
        
def mae_loss(out, targets):
    assert out.size()==targets.size()
    if out.numel()==0:
        return torch.Tensor()
    else:
        assert out.dim()>1, 'MAE loss currently supports Tensors of dimensions > 1'
        losses = torch.abs(out - targets)
        reduce_dims = tuple(list(range(0, len(targets.shape))))
        losses = torch.mean(losses, dim=reduce_dims)
        print("losses: ", losses)
        return losses