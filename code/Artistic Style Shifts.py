import argparse
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import cv2
import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def transfer_style(content_img, style_img):
    # Convert images to YUV color space
    content_yuv = cv2.cvtColor(content_img, cv2.COLOR_BGR2YUV)
    style_yuv = cv2.cvtColor(style_img, cv2.COLOR_BGR2YUV)
    
    # Split YUV channels
    content_y, content_u, content_v = cv2.split(content_yuv)
    style_y, style_u, style_v = cv2.split(style_yuv)
    
    # Apply AdaIN on Y channel
    #y = adaIN(content_y, style_y.mean(), style_y.std())
    #print(u,v,y) 
    # Copy UV channels from the original images
    u = content_u
    v = content_v
    y = style_y    
    # Merge YUV channels
    print(y,u,v)
    yuv = cv2.merge((y, u, v))
    
    # Convert YUV image back to BGR
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    return result

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    print(content.shape[1])
    if(content.shape[1]==3):   
    	content_f = vgg(content)
    	style_f = vgg(style)
    	if interpolation_weights:
        	_, C, H, W = content_f.size()
        	feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        	base_feat = adaptive_instance_normalization(content_f, style_f)
        	for i, w in enumerate(interpolation_weights):
            		feat = feat + w * base_feat[i:i + 1]
        	content_f = content_f[0:1]
    	else:
        	feat = adaptive_instance_normalization(content_f, style_f)
    	feat = feat * alpha + content_f * (1 - alpha)
    	return decoder(feat)
    return None

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='style_output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)
i=0
for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        if(output==None):
           continue        
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        con=Image.open(str(content_path))
        con=np.asarray(con)
        #output=np.array(output)
        #output = output[0].transpose((1, 2, 0))
        #output=transfer_style(con,output)
        #output = output[0].transpose((1, 2, 0))
        #print(output.shape)
        i+=1
        save_image(output, str(i)+str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            if(content.shape[1]!=3):
               os.remove(str(content_path))
               continue
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            if(output==None):
               continue
            output = output.cpu()
            output_name = output_dir / '{:s}{:s}'.format(
                content_path.stem, args.save_ext)
            con=Image.open(str(content_path))
            con = np.asarray(con)
            i+=1
            #output=np.array(output)
            #print(output.shape)
            #output = output[0].transpose((1, 2, 0))
            #output=transfer_style(con, output)
            save_image(output,str(output_name)[:13]+str(i)+ str(output_name)[13:])
