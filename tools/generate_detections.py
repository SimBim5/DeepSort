# vim: expandtab:ts=4:sw=4

import os
import errno
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision import models

##import AP3D 
from deep_sort.models.ResNet import AP3DResNet50
import deep_sort.models.transforms as ST

##import TKP Networks:
from deep_sort.models.ResNet_TKP import ImgResNet50
from deep_sort.models.ResNet_TKP import VidNonLocalResNet50

from deep_sort.models.__init__ import init_model


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape=None):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    if patch_shape is not None:
        image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ResNet50Encoder:

    def __init__(self, pretrained_path=None):
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[:-1]))
        if pretrained_path is not None:
            print("Loading ResNet50Encoder from checkpoint %s" % pretrained_path)
            self.cnn.load_state_dict(torch.load(pretrained_path))
        self.cnn.eval().cuda()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode(self, x):
        with torch.no_grad():
            # use only the last available image
            x = x[-1]
            # apply image transform
            x = self.transform(x)
            # add batch dimension
            x = x.unsqueeze(dim=0)
            # copy tensor to gpu
            x = x.cuda()
            # forward image through backbone
            x = self.cnn(x)
            x = x.view(2048).cpu().numpy()
            return x


class ResNet50AverageEncoder:

    def __init__(self, pretrained_path=None):
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[:-1]))
        if pretrained_path is not None:
            print("Loading ResNet50AverageEncoder from checkpoint %s" % pretrained_path)
            self.cnn.load_state_dict(torch.load(pretrained_path))
        self.cnn.eval().cuda()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((190, 50))
        ])

    def encode(self, x):
        with torch.no_grad():
            # apply image transform
            x = torch.stack([self.transform(y) for y in x])
            # copy tensor to gpu
            x = x.cuda()
            # forward image through backbone
            x = self.cnn(x)
            x = x.view(x.size(0), 2048)
            x = torch.mean(x, dim=0).view(2048).cpu().numpy()
            return x


class AP3DEncoder:

    def __init__(self, pretrained_path=None):
        self.model = AP3DResNet50(625)
        self.transform = ST.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((256, 128), interpolation=3),
        ])
        if pretrained_path is not None:
            print("Loading AP3DEncoder from checkpoint %s" % pretrained_path)
            checkpoint = torch.load(pretrained_path)
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval().cuda()

    def encode(self, x):
        with torch.no_grad():
            x = torch.stack([self.transform(y) for y in x][-3:])
            if x.size(0) in [1, 2]:
                x = torch.stack((3 - (x.size(0) - 1)) * [x])
                x = x.permute(1, 0, 2, 3, 4)
                x = x.squeeze()
            x = x.unsqueeze(dim=0)
            x = x.cuda()
            n, c, f, h, w = x.size()
            assert (n == 1)
            feat = self.model(x)
            feat = feat.mean(1)
            feat = self.model.bn(feat)
            feat = feat.data.squeeze().cpu().numpy()
            return feat


class TKPEncoder:
    ##diesmal 2 Modelle laden: Image Repr. Net. & Video Repr. Net.
    def __init__(self, pretrained_path=None):
        self.model1 = ImgResNet50()
        self.model2 = VidNonLocalResNet50()
        self.transform = ST.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((256, 128), interpolation=3),
        ])
        # laden von weights beider models
        if pretrained_path is not None:
            print("Loading ImgResNet50 from checkpoint %s" % pretrained_path)
            checkpoint = torch.load(pretrained_path)
            self.model1 = init_model(name='img_resnet50')
            self.model1.load_state_dict(checkpoint['img_model_state_dict'])
            self.model1.eval().cuda()
            print("Loading VidNonLocalResNet50 from checkpoint %s" % pretrained_path)
            self.model2 = init_model(name='vid_nonlocalresnet50')
            self.model2.load_state_dict(checkpoint['vid_model_state_dict'])
            self.model2.eval().cuda()

    def encode(self, x):
        with torch.no_grad():
            x = torch.stack([self.transform(y) for y in x][-4:])  ##mind. 4 Bboxen zum Modell
            if x.size(0) in [1, 2, 3]:  ## falls 1, 2, 3 bboxen vorhanden sind:
                x = x[0, :, :, :].unsqueeze(0)
                x = x.cuda()
                feat = self.model1(x)
                feat = feat.data.squeeze().cpu().numpy()
            elif x.size(0) > 3:  ##falls mind. 4 bboxen vorhanden sind:
                x = x.unsqueeze(dim=0)
                n, c, f, h, w = x.size()
                assert (n == 1)
                x = x.permute(0, 2, 1, 3, 4)
                # x = x.squeeze()
                x = x.cuda()
                feat = self.model2(x)
                feat = feat.mean(1)
                feat = feat.data.squeeze().cpu().numpy()
            return feat


def create_box_encoder(model='ResNet50', pretrained_path=None):
    if model == 'ResNet50':
        backbone_cls = ResNet50Encoder
    elif model == "ResNet50Average":
        backbone_cls = ResNet50AverageEncoder
    elif model == 'AP3D':
        backbone_cls = AP3DEncoder
    elif model == 'TKP':
        backbone_cls = TKPEncoder
    else:
        raise Exception('model not found...')
    backbone = backbone_cls(pretrained_path=pretrained_path)
    return backbone


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.
    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
                                "standard MOT detections Directory structure should be the default "
                                "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
                             " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()
