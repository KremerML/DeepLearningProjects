import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()

        self.pad_size = args.prompt_size
        self.image_size = args.image_size

        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, self.image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, self.image_size]))

        self.pad_left = nn.Parameter(torch.randn([1, 3, self.image_size - self.pad_size*2, self.pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, self.image_size - self.pad_size*2, self.pad_size]))


    def forward(self, x):
        inner_upper_bound = self.pad_size
        inner_lower_bound = self.image_size - self.pad_size
        outer_lower_bound = self.image_size

        prompt = torch.zeros_like(x)
        prompt[:, :, 0:inner_upper_bound, :] = self.pad_up
        prompt[:, :, inner_lower_bound:outer_lower_bound, :] = self.pad_down

        prompt[:, :, inner_upper_bound:inner_lower_bound, 0:inner_upper_bound] = self.pad_left
        prompt[:, :, inner_upper_bound:inner_lower_bound, inner_lower_bound:outer_lower_bound] = self.pad_right

        return x + prompt
        


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
    
        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        

        # self.device = args.device
        self.image_size = args.image_size
        self.prompt_size = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.prompt_size, self.prompt_size]))


    def forward(self, x):
        prompt = torch.zeros_like(x)
        prompt[:, :, 0:self.prompt_size, 0:self.prompt_size] = self.patch

        return x + prompt
    

class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        # self.device = args.device
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))
        

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize])
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt
        

class CoverImagePrompter(nn.Module):
    """
    Defines visual-prompt that covers the entire image.
    """
    def __init__(self, args):
        super(CoverImagePrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        # assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        # self.device = args.device
        self.image_size = args.image_size
        self.prompt_size = args.image_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.prompt_size, self.prompt_size]))
        

    def forward(self, x):
        prompt = torch.zeros_like(x)
        prompt[:, :, 0:self.prompt_size, 0:self.prompt_size] = self.patch

        return x + prompt

        

