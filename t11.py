import matplotlib.pyplot as plt
import torchvision.transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import torch

def save_images(original_image, watermark_image, noise_image, epoch):
    """
    Save the given images as a single image file.
    """
    batch_size = original_image.shape[0]
    rows = 3
    cols = batch_size

    image_size = original_image.shape[-1]


    # Normalize the watermark image and noise image if their pixel values are not in [0, 1]
    watermark_image_max = torch.max(watermark_image)
    noise_image_max = torch.max(noise_image)
    if watermark_image_max > 1 or noise_image_max > 1:
        watermark_image = F.sigmoid(watermark_image)
        noise_image = F.sigmoid(noise_image)

    images = [original_image, watermark_image, noise_image]
    # Create a new figure and plot the images
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    for col in range(cols):
        for row in range(rows):
            image = images[row][col]
            title = ''
            if row == 0:
                title = 'Original Image'
            elif row == 1:
                title = 'Watermark Image'
            elif row == 2:
                title = 'Noise Image'
            # print(image)
            axs[row][col].imshow(to_pil_image(image.cpu()))
            axs[row][col].axis('off')
            axs[row][col].set_title(title)

    # Save the figure to a file
    filename = f"result_epoch_{epoch}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)


from PIL import Image
if __name__ == '__main__':
    p1 = Image.open('1.png')
    p1 = torchvision.transforms.PILToTensor()(p1).div(255).expand(2, -1, -1, -1)
    p2 = p1 + torch.randn_like(p1)
    p3 = p1 + torch.randn_like(p1)
    save_images(p1, p2, p3, 10)
    print(1)