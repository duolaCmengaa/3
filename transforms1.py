import random
import math
import numbers

import cv2
import numpy as np

import torch

class Compose:
    """Composes several transforms together.

    Args:
        transforms(list of 'Transform' object): list of transforms to compose

    """    

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):

        for trans in self.transforms:
            img = trans(img)
        
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToCVImage:
    """Convert an Opencv image to a 3 channel uint8 image
    """

    def __call__(self, image):
        """
        Args:
            image (numpy array): Image to be converted to 32-bit floating point
        
        Returns:
            image (numpy array): Converted Image
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image = image.astype('uint8')
            
        return image


class RandomResizedCrop:
    """Randomly crop a rectangle region whose aspect ratio is randomly sampled 
    in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped
    region into a 224-by-224 square image.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR: 
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'):

        self.methods={
            "area":cv2.INTER_AREA, 
            "nearest":cv2.INTER_NEAREST, 
            "linear" : cv2.INTER_LINEAR, 
            "cubic" : cv2.INTER_CUBIC, 
            "lanczos4" : cv2.INTER_LANCZOS4
        }

        self.size = (size, size)
        self.interpolation = self.methods[interpolation]
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        h, w, _ = img.shape

        area = w * h

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            target_ratio = random.uniform(*self.ratio) 

            output_h = int(round(math.sqrt(target_area * target_ratio)))
            output_w = int(round(math.sqrt(target_area / target_ratio))) 

            if random.random() < 0.5:
                output_w, output_h = output_h, output_w 

            if output_w <= w and output_h <= h:
                topleft_x = random.randint(0, w - output_w)
                topleft_y = random.randint(0, h - output_h)
                break

        if output_w > w or output_h > h:
            output_w = min(w, h)
            output_h = output_w
            topleft_x = random.randint(0, w - output_w) 
            topleft_y = random.randint(0, h - output_w)

        cropped = img[topleft_y : topleft_y + output_h, topleft_x : topleft_x + output_w]

        resized = cv2.resize(cropped, self.size, interpolation=self.interpolation)

        return resized
    
    def __repr__(self):
        for name, inter in self.methods.items():
            if inter == self.interpolation:
                inter_name = name

        interpolate_str = inter_name
        format_str = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_str += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_str += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_str += ', interpolation={0})'.format(interpolate_str)

        return format_str


class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)
        
        return img

class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch 
    float tensor (c, h, w) ranged from 0 to 1
    """

    def __call__(self, img):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]
        
        Returns:
            a pytorch tensor
        """
        #convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        return img

class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    
    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, img):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """        
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        if not self.inplace:
            img = img.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img

class Resize:

    def __init__(self, resized=256, interpolation='linear'):

        methods = {
            "area":cv2.INTER_AREA, 
            "nearest":cv2.INTER_NEAREST, 
            "linear" : cv2.INTER_LINEAR, 
            "cubic" : cv2.INTER_CUBIC, 
            "lanczos4" : cv2.INTER_LANCZOS4
        }
        self.interpolation = methods[interpolation]

        if isinstance(resized, numbers.Number):
            resized = (resized, resized)
        
        self.resized = resized

    def __call__(self, img):


        img = cv2.resize(img, self.resized, interpolation=self.interpolation)

        return img
class CenterCrop:
    """Crops the given image at the center.

    Args:
        size: expected output size of each edge. If size is a sequence like 
              (h, w), output size will be matched to this. If size is an int, 
              a square crop of (size, size) is made.
    """
    
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be center cropped.
        
        Returns:
            img (numpy array): Center cropped image.
        """
        h, w, _ = img.shape
        crop_h, crop_w = self.size

        if crop_w > w or crop_h > h:
            raise ValueError("Requested crop size {} is bigger than input size {}.".format(self.size, (h, w)))

        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2

        cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
        
        return cropped

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    

class RandomPerspective:
    """Perform perspective transformation of input opencv image with the given distortion scale.

    Args:
        distortion_scale: it determines the amount of distortion in perspective transformation
        p: probability of the image being transformed
        interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, distortion_scale, p=0.5, interpolation='linear'):
        self.distortion_scale = distortion_scale
        self.p = p
        self.methods = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        self.interpolation = self.methods[interpolation]

    def __call__(self, img):
        if random.random() < self.p:
            height, width = img.shape[:2]

            # Four pairs of corresponding points
            pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            pts2 = np.float32([[random.uniform(-self.distortion_scale, self.distortion_scale) * width, 
                                random.uniform(-self.distortion_scale, self.distortion_scale) * height],
                               [width - random.uniform(-self.distortion_scale, self.distortion_scale) * width, 
                                random.uniform(-self.distortion_scale, self.distortion_scale) * height],
                               [random.uniform(-self.distortion_scale, self.distortion_scale) * width, 
                                height - random.uniform(-self.distortion_scale, self.distortion_scale) * height],
                               [width - random.uniform(-self.distortion_scale, self.distortion_scale) * width, 
                                height - random.uniform(-self.distortion_scale, self.distortion_scale) * height]])

            # Compute the perspective transform matrix and apply it
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (width, height), flags=self.interpolation)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(distortion_scale={}, p={}, interpolation={})'.format(self.distortion_scale, self.p, self.interpolation)
class RandomRotation:
    """随机旋转图像。

    参数：
        degrees (序列或浮点数或整数): 选择的角度范围。如果 degrees 是一个数字而不是序列,例如 (min, max)，则角度范围为 (-degrees, +degrees)。
        interpolation (字符串): 插值方法。默认为 'linear'。
        expand (布尔值): 可选的扩展标志。如果为 true,则扩展输出图像以包含整个旋转图像。如果为 false 或省略，则输出图像与输入图像大小相同。
        center (元组): 可选的旋转中心。原点为左上角。默认是图像中心。
        fill (整数或元组): 可选的填充颜色，用于填充输出图像中的变换区域。默认为 0(黑色)。
    """

    def __init__(self, degrees, interpolation='linear', expand=False, center=None, fill=0):
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("如果 degrees 是一个单一数字，它必须是非负的。")
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        
        self.methods = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        self.interpolation = self.methods[interpolation]
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        h, w = img.shape[:2]

        if self.center is None:
            center = (w / 2, h / 2)
        else:
            center = self.center

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        if self.expand:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])

            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            matrix[0, 2] += (nW / 2) - center[0]
            matrix[1, 2] += (nH / 2) - center[1]

            img = cv2.warpAffine(img, matrix, (nW, nH), flags=self.interpolation, borderValue=self.fill)
        else:
            img = cv2.warpAffine(img, matrix, (w, h), flags=self.interpolation, borderValue=self.fill)
        
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={}, interpolation={}, expand={}, center={}, fill={})'.format(
            self.degrees, self.interpolation, self.expand, self.center, self.fill)
