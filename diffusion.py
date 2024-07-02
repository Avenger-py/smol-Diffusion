import torch
from utils import show_tensor_image, show_any_images
import matplotlib.pyplot as plt

class Diffusion():
    def __init__(self, timesteps, img_size, device):
        self.T = timesteps
        self.img_size = img_size
        self.device = device
        self.betas = self.cosine_schedule(num_timesteps=self.T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def cosine_schedule(self, num_timesteps, s=0.008):
        def f(t):
            return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
        x = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.999)
        return betas
    
    def sample_t_index(self, val, t):
        out = val[t]
        return out[:, None, None, None]
    
    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sample_t_index(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = self.sample_t_index(self.sqrt_one_minus_alphas_cumprod, t)
        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)
    
    def vis_forward_diffusion(self, image):
        plt.figure(figsize=(15,2))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.T/num_images)

        for idx in range(0, self.T, stepsize):
            t = torch.Tensor([idx]).type(torch.int64)
            plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
            img, noise = self.forward_diffusion_sample(image, t)
            show_tensor_image(img.cpu())
        plt.show()

    def test_sampling(self, eps, x, t):
        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        recip_sqrt_alpha_t = 1 / torch.sqrt(alpha_t)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        if t > 1:
            noise = torch.randn_like(x)
        else:
            noise = 0
        
        x =  recip_sqrt_alpha_t * (x - beta_t * eps / sqrt_one_minus_alphas_cumprod_t) + torch.sqrt(beta_t) * noise
        return x

    def test_reverse_diffusion(self, t_target, img):
        assert t_target.shape[0] == 1
        assert t_target.item() < self.T

        # Sequential forward diffusion
        epsilon_list = []
        img_t_target = None
        for i in range(t_target.item()):
            t_inp = torch.full((1,), i, device=self.device, dtype=torch.long)
            _, ep = self.forward_diffusion_sample(img, t_inp)
            epsilon_list.append(ep)
            if i==t_target.item()-1:
                img_t_target = _

        # Sequential denoising
        img_size = self.img_size
        img_denoised = torch.randn((1, 3, img_size, img_size), device=self.device)
        for i in range(t_target.item())[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img_denoised = self.test_sampling(epsilon_list[i], img_denoised.to(self.device), t)
            img_denoised = torch.clamp(img, -1.0, 1.0)
        
        # Compare results
        show_any_images([img, img_t_target, img_denoised], ['Original', 'Noisy', 'Denoised'], cols=3)
    
    @torch.no_grad()
    def sample_timestep(self, model, x, t):

        betas_t = self.sample_t_index(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = self.sample_t_index(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip_alphas_t = self.sample_t_index(self.sqrt_recip_alphas, t)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t.item() == 0:
            noise = 0
        else:
            noise = torch.randn_like(x)

        return model_mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample_plot_image(self, model):
        img_size = self.img_size
        img = torch.randn((1, 3, img_size, img_size), device=self.device)
        plt.figure(figsize=(15,2))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.T/num_images)

        for i in range(0,self.T)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(model, img, t)
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                show_tensor_image(img.detach().cpu())
        plt.show()