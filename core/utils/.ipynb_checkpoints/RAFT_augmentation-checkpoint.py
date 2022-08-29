# -*- coding: utf-8 -*-
# +
# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

CLIP_MAX = 1e3
DEFAULT_ERASER_BOUNDS = (50, 100)


class raftAugment(tf.keras.layers.Layer):
  """Augment object for RAFT"""

  def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5):
    super(raftAugment, self).__init__()

    
    self.crop_size = crop_size

    self.brightness = (0.6, 1.4) # 亮度范围 0.6 0.7 0.7 img1  1.2 1.2 1.2 img2
    self.contrast = (0.6, 1.4)   # 对比度范围
    self.saturation = (0.6, 1.4) # 饱和度范围
    self.hue = 0.5 / 3.14        # 调整色调

    self.asymmetric_color_aug_prob = 0.2 # 20%的几率进行非对称颜色增强(augment_color),两张图片分布随机增强, 80%的几率进行对称颜色增强
    self.spatial_aug_prob = 0.8 # 80%的几率进行空间增强(random_scale), 挑出图片的一部分, 进行双线性插值, 20%不变
    self.eraser_aug_prob = 0.5  # 50%的几率进行擦除增强(eraser_transform), 把图片的一部分替换为颜色均值

    # 以下五个为空间增强的参数
    self.min_scale = min_scale
    self.max_scale = max_scale
    self.max_stretch = 0.2
    self.stretch_prob = 0.8
    self.margin = 20

  def augment_color(self, images):
    brightness_scale = tf.random.uniform([],
                                         self.brightness[0],
                                         self.brightness[1],
                                         dtype=tf.float32)
    images = images * brightness_scale
    # images = tf.clip_by_value(images, 0, 1) # float limits
    images = tf.image.random_contrast(images, self.contrast[0],
                                      self.contrast[1])
    # images = tf.clip_by_value(images, 0, 1) # float limits
    images = tf.image.random_saturation(images, self.saturation[0],
                                        self.saturation[1])
    # images = tf.clip_by_value(images, 0, 1) # float limits
    images = tf.image.random_hue(images, self.hue)
    images = tf.clip_by_value(images, 0, 1)  # float limits
    return images

  def color_transform(self, img1, img2):
    pred = tf.random.uniform([]) < self.asymmetric_color_aug_prob
    def true_fn(img1, img2):
      img1 = self.augment_color(img1)
      img2 = self.augment_color(img2)
      return [img1, img2]
    def false_fn(img1, img2):
      imgs = tf.concat((img1, img2), axis=0)
      imgs = self.augment_color(imgs)
      return tf.split(imgs, num_or_size_splits=2)

    return tf.cond(pred, lambda: true_fn(img1, img2),
                   lambda: false_fn(img1, img2))

  def eraser_transform(self, img1, img2, bounds=DEFAULT_ERASER_BOUNDS):
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    pred = tf.random.uniform([]) < self.eraser_aug_prob
    def true_fn(img1, img2):
      mean_color = tf.reduce_mean(tf.reshape(img2, (-1, 3)), axis=0)
      mean_color = tf.expand_dims(tf.expand_dims(mean_color, axis=0), axis=0)
      def body(var_img, mean_color):
        x0 = tf.random.uniform([], 0, wd, dtype=tf.int32)
        y0 = tf.random.uniform([], 0, ht, dtype=tf.int32)
        dx = tf.random.uniform([], bounds[0], bounds[1], dtype=tf.int32)
        dy = tf.random.uniform([], bounds[0], bounds[1], dtype=tf.int32)
        x = tf.range(wd)
        x_mask = (x0 <= x) & (x < x0+dx)
        y = tf.range(ht)
        y_mask = (y0 <= y) & (y < y0+dy)
        mask = x_mask & y_mask[:, tf.newaxis]
        mask = tf.cast(mask[:, :, tf.newaxis], img1.dtype)
        mean_slice = tf.tile(mean_color, multiples=[ht, wd, 1])
        result = var_img * (1 - mask) + mean_slice * mask
        return result
      max_num = tf.random.uniform([], 1, 3, dtype=tf.int32)
      img2 = body(img2, mean_color)
      img2 = tf.cond(2 <= max_num, lambda: body(img2, mean_color), lambda: img2)
      return img1, img2
    def false_fn(img1, img2):
      return img1, img2

    return tf.cond(pred, lambda: true_fn(img1, img2),
                   lambda: false_fn(img1, img2))

  def random_vertical_flip(self, img1, img2, flow, prob=0.1):
    pred = tf.random.uniform([]) < prob
    def true_fn(img1, img2, flow):
      img1 = tf.image.flip_up_down(img1)
      img2 = tf.image.flip_up_down(img2)
      flow = tf.image.flip_up_down(flow) * [1.0, -1.0]
      return img1, img2, flow
    def false_fn(img1, img2, flow):
      return img1, img2, flow
    return tf.cond(pred,
                   lambda: true_fn(img1, img2, flow),
                   lambda: false_fn(img1, img2, flow))

  def random_horizontal_flip(self, img1, img2, flow, prob=0.5):
    pred = tf.random.uniform([]) < prob
    def true_fn(img1, img2, flow):
      img1 = tf.image.flip_left_right(img1)
      img2 = tf.image.flip_left_right(img2)
      flow = tf.image.flip_left_right(flow) * [-1.0, 1.0]
      return img1, img2, flow
    def false_fn(img1, img2, flow):
      return img1, img2, flow
    return tf.cond(pred,
                   lambda: true_fn(img1, img2, flow),
                   lambda: false_fn(img1, img2, flow))

  def random_scale(self, img1, img2, flow, scale_x, scale_y):
    pred = tf.random.uniform([]) < self.spatial_aug_prob
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    def true_fn(img1, img2, flow, scale_x, scale_y):
      # rescale the images
      new_ht = scale_x * tf.cast(ht, dtype=tf.float32)
      new_wd = scale_y * tf.cast(wd, dtype=tf.float32)
      new_shape = tf.cast(tf.concat([new_ht, new_wd], axis=0), dtype=tf.int32)
      img1 = tf.compat.v1.image.resize(
          img1,
          new_shape,
          tf.compat.v1.image.ResizeMethod.BILINEAR,
          align_corners=True)
      img2 = tf.compat.v1.image.resize(
          img2,
          new_shape,
          tf.compat.v1.image.ResizeMethod.BILINEAR,
          align_corners=True)
      flow = tf.compat.v1.image.resize(
          flow,
          new_shape,
          tf.compat.v1.image.ResizeMethod.BILINEAR,
          align_corners=True)

      flow = flow * tf.expand_dims(
          tf.expand_dims(tf.concat([scale_x, scale_y], axis=0), axis=0), axis=0)
      return img1, img2, flow

    def false_fn(img1, img2, flow):
      return img1, img2, flow
    return tf.cond(pred,
                   lambda: true_fn(img1, img2, flow, scale_x, scale_y),
                   lambda: false_fn(img1, img2, flow))

  def spatial_transform(self, img1, img2, flow):
    # randomly sample scale
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    min_scale = tf.math.maximum(
        (self.crop_size[0] + 1) / ht,
        (self.crop_size[1] + 1) / wd)

    max_scale = self.max_scale
    min_scale = tf.math.maximum(min_scale, self.min_scale)

    scale = 2 ** tf.random.uniform([], self.min_scale, self.max_scale)
    scale_x = scale
    scale_y = scale
    pred = tf.random.uniform([]) < self.stretch_prob
    def true_fn(scale_x, scale_y):
      scale_x *= 2 ** tf.random.uniform([], -self.max_stretch, self.max_stretch)
      scale_y *= 2 ** tf.random.uniform([], -self.max_stretch, self.max_stretch)
      return tf.stack((scale_x, scale_y), axis=0)
    def false_fn(scale_x, scale_y):
      return tf.stack((scale_x, scale_y), axis=0)
    scales = tf.cond(pred,
                     lambda: true_fn(scale_x, scale_y),
                     lambda: false_fn(scale_x, scale_y))
    scale_x, scale_y = tf.split(scales, num_or_size_splits=2)

    clip_max = tf.cast(CLIP_MAX, dtype=tf.float32)
    min_scale = tf.cast(min_scale, dtype=tf.float32)
    scale_x = tf.clip_by_value(scale_x, min_scale, clip_max)
    scale_y = tf.clip_by_value(scale_y, min_scale, clip_max)

    img1, img2, flow = self.random_scale(img1, img2, flow, scale_x, scale_y)

    # random flips
    img1, img2, flow = self.random_horizontal_flip(img1, img2, flow, prob=0.5)
    img1, img2, flow = self.random_vertical_flip(img1, img2, flow, prob=0.1)

    # clip_by_value
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    y0 = tf.random.uniform([],
                           -self.margin,
                           ht - self.crop_size[0] + self.margin,
                           dtype=tf.int32)
    x0 = tf.random.uniform([],
                           -self.margin,
                           wd - self.crop_size[1] + self.margin,
                           dtype=tf.int32)

    y0 = tf.clip_by_value(y0, 0, ht - self.crop_size[0])
    x0 = tf.clip_by_value(x0, 0, wd - self.crop_size[1])

    # crop
    img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]:, :]
    img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]:, :]
    flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]:, :]

    return img1, img2, flow

  def __call__(self, images, flow):
#     images = tf.convert_to_tensor(images.astype("float32")) / 255.0
#     flow = tf.convert_to_tensor(flow.astype("float32"))

#     images = (images + 1) / 2.0  # switch from [-1,1] to [0,1]
#     with tf.device('/GPU'):
    img1, img2 = tf.unstack(images, num=2)
    img1, img2 = self.color_transform(img1, img2)
    img1, img2 = self.eraser_transform(img1, img2)
    img1, img2, flow = self.spatial_transform(img1, img2, flow)
    images = tf.stack((img1, img2), axis=0)
    images = tf.ensure_shape(images,
                             (2, self.crop_size[0], self.crop_size[1], 3))
    flow = tf.ensure_shape(flow, (self.crop_size[0], self.crop_size[1], 2))

    images = 2 * images - 1  # switch from [0,1] to [-1,1]
    return images, flow


def apply(element, aug_params):
  crop_size = (aug_params.crop_height, aug_params.crop_width)
  min_scale = aug_params.min_scale
  max_scale = aug_params.max_scale
  aug = raftAugment(crop_size=crop_size, min_scale=min_scale, max_scale=max_scale)
  images, flow = aug(element['inputs'], element['label'])
  return {'inputs': images, 'label':flow}



# -

if __name__ == "__main__":
#     images, flow = aug(tf.random.uniform([2,512,512,3]),tf.random.uniform([512,512,2]))
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import cv2
    import numpy as np

    def readFlowKITTI(filename):
        flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
        flow = flow[:,:,::-1].astype(np.float32)
        flow, valid = flow[:, :, :2], flow[:, :, 2]
        flow = (flow - 2**15) / 64.0
        return flow, valid    

    def renderFlowImg(flow, *, maxFlow=-1, style='mpi'):
        MVY_PLANE   = 0
        MVX_PLANE   = 1
        assert style in ['kitti-c++', 'mpi'], 'unknown flow rendering style: {}'.format(style)
        assert len(flow.shape) == 3
        mvx = flow[:,:,MVX_PLANE]
        mvy = flow[:,:,MVY_PLANE]
        magnitude = np.sqrt(mvx ** 2 + mvy ** 2)
        direction = np.arctan2(mvy, mvx)
        if maxFlow < 0:
            if style == 'kitti-c++':
                maxFlow = np.fmax(magnitude.max(), 1.0)
            else:
                # mpi style
                maxFlow = 1.2 * max(np.fabs(mvx).max(), np.fabs(mvy).max())
                maxFlow = np.fmax(maxFlow, 1.0)
            print('maxFlow = {:.1f}'.format(maxFlow))
        h = np.fmod(direction / (2 * np.pi) + 1.0, 1.0) * 360
        assert h.min() >= 0
        if style == 'kitti-c++':
            """KITTI C++ hsv2rgb has bug. Here tries to mimic the final result"""
            s = np.clip(magnitude * 300 / maxFlow, 0, 1.0)
            v = np.clip(magnitude * 8   / maxFlow, 0, 1.0)
        else:
            # mpi style
            n = 8
            s = np.clip(magnitude * n / maxFlow, 0, 1.0)
            v = np.clip(n - s, 0, 1.0)
        img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
    #     img[~valid] = 0
    #     return np.uint8(img * 255)[:,:,[1,2,0]]
        return np.uint8(img * 255)
    
    
    def drawConcat(images, flow):
        nrow, ncol = 1, 3
        fig = plt.figure(figsize=(ncol*5, nrow*5))

        gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.0, hspace=0.0, 
             top=(1.-0.5/(nrow+1))*5, bottom=5*(0.5/(nrow+1)), 
             left=(0.5/(ncol+1)), right=(1-0.5/(ncol+1))) 

        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[0,2])

        ax0.imshow(images[0])
        ax0.axis("off")
        ax1.imshow(images[1])
        ax1.axis("off")
        ax2.imshow(renderFlowImg(flow))
        ax2.axis("off")
        plt.show()


if __name__ == "__main__":
    image1 = cv2.imread("/raft_demo/MOVI/train/images/000000/00.png")
    image2 = cv2.imread("/raft_demo/MOVI/train/images/000000/01.png")
    flow,valid = readFlowKITTI("/raft_demo/MOVI/train/forwardflow/000000/00.png")
    
    images = tf.convert_to_tensor([image1.astype("float32")/255.0,image2.astype("float32")/255.0])
    flow = tf.convert_to_tensor(flow.astype("float32"))
    drawConcat(images, flow)


if __name__ == "__main__":
    aug = raftAugment(crop_size=(320, 448))
    imagesAug, flowAug = aug(images,flow)
    drawConcat(imagesAug, flowAug)


