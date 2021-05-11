import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.notebook import tqdm
import scipy.io as sio
import os
import imageio
from sklearn.decomposition import FastICA, PCA

IMAGE_URLS = ["http://www.rctn.org/bruno/sparsenet/IMAGES.mat", "http://www.rctn.org/bruno/sparsenet/IMAGES_RAW.mat", "https://dz2cdn1.dzone.com/storage/temp/3542733-printed-circuit-boards.jpg"]

print("Downloading O&F images and circuit board...")
for file_url in IMAGE_URLS:
    file_name = os.path.basename(file_url)
    if os.path.exists(file_name):
        print(f"{file_url} has already been downloaded.")
    else:
        print(f"Starting to download {file_url}...")
        os.system(f"wget {file_url}")
        print(f"...download complete.")
print("...all downloads complete.")

print("Importing natural_imgs, natural_imgs_raw, circuit_imgs_raw.") 
mat_images = sio.loadmat('IMAGES.mat')
natural_imgs = mat_images['IMAGES']
mat_images_raw = sio.loadmat('IMAGES_RAW.mat')
natural_imgs_raw = mat_images_raw['IMAGESr']
circuit_imgs_raw = imageio.imread("3542733-printed-circuit-boards.jpg")

class OlshausenField1996Model:
    def __init__(self, num_inputs, num_units, batch_size,
                    thresh_type="soft",
                    nt_max=1000, eps=1e-2,
                    lr_r=1e-2, lr_Phi=1e-2, lmda=5e-3):
        self.lr_r = lr_r # learning rate of r
        self.lr_Phi = lr_Phi # learning rate of Phi
        self.lmda = lmda # regularization parameter

        self.nt_max = nt_max # Maximum number of simulation time
        self.eps = eps  # small value which determines convergence
        
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.batch_size = batch_size

        assert thresh_type in ["soft", "ln"]
        self.thresh_type = thresh_type
        if self.thresh_type == "soft":
            self._spasity_func = lambda x: np.abs(x)
            self._thresh_func = self.soft_thresholding_func
        elif self.thesh_type == "ln":
            self._spasity_func = lambda x: np.ln(1 + x**2)
            self._thresh_func = self.ln_thresholding_func
        elif self.thesh_type == "cauchy":
            self._spasity_func = lambda x: np.abs(x)
            self._thresh_func = self.cauchy_thresholding_func

        # Weights
        Phi = np.random.randn(self.num_inputs, self.num_units).astype(np.float32)
        self.Phi = Phi * np.sqrt(1/self.num_units)

        # activity of neurons
        self.r = np.zeros((self.batch_size, self.num_units))
    
    def initialize_states(self):
        self.r = np.zeros((self.batch_size, self.num_units))
        
    def normalize_rows(self):
        self.Phi = self.Phi / np.maximum(np.linalg.norm(self.Phi, ord=2, axis=0, keepdims=True), 1e-8)

    # thresholding function of S(x)=|x|
    def soft_thresholding_func(self, x, lmda):
        return np.maximum(x - lmda, 0) - np.maximum(-x - lmda, 0)

    # thresholding function of S(x)=ln(1+x^2)
    def ln_thresholding_func(self, x, lmda):
        f = 9*lmda*x - 2*np.power(x, 3) - 18*x
        g = 3*lmda - np.square(x) + 3
        h = np.cbrt(np.sqrt(np.square(f) + 4*np.power(g, 3)) + f)
        two_croot = np.cbrt(2) # cubic root of two
        return (1/3)*(x - h / two_croot + two_croot*g / (1e-8+h))

    # thresholding function https://arxiv.org/abs/2003.12507
    def cauchy_thresholding_func(self, x, lmda):
        f = 0.5*(x + np.sqrt(np.maximum(x**2 - lmda,0)))
        g = 0.5*(x - np.sqrt(np.maximum(x**2 - lmda,0)))
        return f*(x>=lmda) + g*(x<=-lmda) 

    def calculate_error(self, inputs):
        error = inputs - self.r @ self.Phi.T
        return(error)

    def calculate_total_error(self, error):
        recon_error = np.mean(error**2)
        sparsity_r = self.lmda*np.mean(self._spasity_func(self.r)) 
        return(recon_error + sparsity_r)

    def update_r(self, inputs):
        error = self.calculate_error(inputs)
        r = self.r + self.lr_r * error @ self.Phi
        self.r = self._thresh_func(r, self.lmda)
        return(error)

    def update_Phi(self, inputs):
        error = self.calculate_error(inputs)
        dPhi = error.T @ self.r
        self.Phi += self.lr_Phi * dPhi
        return(error)
    
    def train(self, inputs):
        self.initialize_states() # Reset states
        self.normalize_rows() # Normalize weights
        
        # Input an image patch until latent variables are converged 
        r_tm1 = self.r # set previous r (t minus 1)
        for t in range(self.nt_max):
            # Update r without updating weights 
            error = self.update_r(inputs)
            dr = self.r - r_tm1 

            # Compute norm of r
            dr_norm = np.linalg.norm(dr, ord=2) / (self.eps + np.linalg.norm(r_tm1, ord=2))
            r_tm1 = self.r # update r_tm1
            
            # Check convergence of r, then update weights
            if dr_norm < self.eps:
                error = self.update_r(inputs)
                error = self.update_Phi(inputs)
                break
            
            # If failure to convergence, break and print error
            if t >= self.nt_max-2: 
                print("Error at patch:", iter_)
                print(dr_norm)
                break
        return(error)

def generate_patches(input_images, patch_size, batch_size):
    H, W, num_images = input_images.shape
    
    # Set the coordinates of the upper left corner of random image sz X sz image clips
    x0 = np.random.randint(0, W-patch_size, batch_size)
    y0 = np.random.randint(0, H-patch_size, batch_size)

    # Generating inputs
    patches_list = []
    for i in range(batch_size):        
        idx = np.random.randint(0, num_images)
        img = input_images[:, :, idx]
        clip = img[y0[i]:y0[i]+patch_size, x0[i]:x0[i]+patch_size].flatten()
        patches_list.append(clip - np.mean(clip))
        
    patches = np.array(patches_list) # Input image patches
    return(patches)

def train_OlshausenField1996Model(image_set, patch_size=16, num_units=100, num_iter=500, batch_size=250):
    
    # Define model
    model = OlshausenField1996Model(num_inputs=patch_size**2, num_units=num_units, batch_size=batch_size)

    # Run simulation
    error_list = [] # List to save errors
    for iter_ in tqdm(range(num_iter)):
        patches = generate_patches(image_set, patch_size, batch_size) # Generating image patches
        error = model.train(patches) # train model with patches 

        error_list.append(model.calculate_total_error(error))
        # Print moving average error
        if iter_ % 100 == 99:  
            print("iter: "+str(iter_+1)+"/"+str(num_iter)+", Moving error:",
                np.mean(error_list[iter_-99:iter_]))
    
    return model, error_list

def plot_receptive_fields(fields, c=10, title="Receptive Fields"):
    num_units, num_inputs = fields.shape
    patch_size = int(np.sqrt(num_inputs))
    r = num_units // c

    fig = plt.figure(figsize=(6, .6*(r+4)))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in tqdm(range(num_units)):
        plt.subplot(r, c, i+1)
        plt.imshow(np.reshape(fields[i], (patch_size, patch_size)), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.9)