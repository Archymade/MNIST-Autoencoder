import matplotlib.pyplot as plt
from jupyterthemes import jtplot
from datetime import datetime

def generate_and_save_images(model, sample, folder, epoch, style = 'gruvboxd', 
                             variational = False, save_image = False):
    
    ''' Generate images via autoencoder, with options to save. '''
    
    if style != 'normal':
        jtplot.style(style)
    
    else:
        import seaborn as sns
        sns.set()
    
    if variational:
        m, l = model.encode(sample)
        z = model.reparametrize(m, l)
        z = model.decode(z)
    else:
        z = model.encode(sample)
        z = model.decode(z)
    
    plt.figure(figsize = (12, 7))
    
    for i in range(z.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.squeeze(z[i]), cmap = 'gray')
    
    plt.tight_layout()
        
    if save_image:
        plt.savefig(f'{folder}/Auto-generated MNIST Image for Epoch [{epoch}]-{datetime.now()}.png', dpi = 300)
    
    plt.show(); plt.close('all')