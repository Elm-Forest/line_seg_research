import torch

from models.deep_hough_transform.dht_module.dht_func import C_dht
from utils.util import create_simple_line_image

if __name__ == '__main__':
    dht = C_dht(180, 200).cuda()
    img = create_simple_line_image().cuda().double()
    img = img.cuda().requires_grad_(True)
    test = torch.autograd.gradcheck(
        dht,
        img,
        eps=1e-6,
        atol=1e-4,
        check_undefined_grad=False
    )

    print("DHT Grad Check Result: ", test)

# # check float16
# if __name__ == '__main__':
#     dht = C_dht(180, 200).cuda()
#     img = create_simple_line_image().cuda().to(torch.float16)
#     o = dht(img)
