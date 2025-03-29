import torch

from models.deep_hough_transform.dht_module.dht_func import C_dht
from models.deep_hough_transform.idht_module.idht_func import C_idht
from utils.util import create_simple_line_image

if __name__ == '__main__':
    dht = C_dht(180, 200).cuda()
    idht = C_idht(180, 200, 50, 50).cuda()
    img = create_simple_line_image().cuda().double()
    hm = dht(img)
    hm = hm.cuda().requires_grad_(True)
    test = torch.autograd.gradcheck(
        idht,
        hm,
        eps=1e-6,
        atol=1e-4,
        check_undefined_grad=False
    )

    print("IDHT Grad Check Result: ", test)
