# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset, BaseDataset_2
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        if "interval" in data and kwargs["augment"]:
            interval = data["interval"]
        else:
            interval = 1
            
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(interval=interval, *args, **kwargs)
        
            
    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.img_path).with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def img2label_paths(self,img_paths):
        # sa, sb = f'{self.images_dir}', f'{self.labels_dir}'  # /images/, /labels/ substrings
        return [path.replace(sa, sb).split('.')[0]+'.txt' for path, sa, sb  in zip(img_paths, self.images_dir, self.labels_dir)]
    
    def get_labels_new(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = self.img2label_paths(self.im_files)
        # cache_path = Path(combine_unique_folders([os.path.splitext(p)[0] for p in self.img_path])).with_suffix('.cache')
        # cache_path = Path(self.img_path).with_suffix(".cache")
        cache_path = Path(self.im_files[0]).with_suffix(".cache") if isinstance(self.img_path, list) else Path(self.img_path).with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels
    
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max([len(s) for s in segments])
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

import os
import numpy as np
import math
from copy import deepcopy
import random
import time,json
import re
import torch.distributed as dist

def combine_unique_folders(paths):
    folder_name_ = [path.split(os.sep) for path in paths]
    folder_names = [folder for path in folder_name_ for folder in path if folder]
    # 使用 set 去重
    unique_folder_names = list(sorted(set(folder_names)))
    # 将独特的文件夹名称用 "_" 连接起来
    combined_string = '_'.join(unique_folder_names)
    return os.path.join(list(paths)[0],combined_string)


class YOLOVideoDataset(BaseDataset_2):
    """

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)
        self.match_number = self.data["match_number"]
        self.interval = self.data["interval"]
        self.interval_mode = self.data["interval_mode"]
        self.videos = self.data.get("videos", True)
        
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
        self.im_frame_matching(self.im_files)
        
        self.length = 1 #epoch 0保证正常初始化dataset
        self.epoch = 0
        self.sub_videos, self.sampler_indices, self.index_mapping_frameinfo  = self.split_sub_videos(self.interval, self.length, self.world_size, is_training = self.augment)
        self.cnt_use = {}
        
    def im_frame_matching(self, im_files):
        # Create a dictionary that groups images by video name
        video_image_dict = {}
        self.videos_info = []
        
        for i, image_path in enumerate(im_files):
            video_name = os.path.basename(os.path.dirname(image_path))
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if self.videos:
                frame_num_string = image_name.split('_')[-1]  # Assuming the video name is the first part of the filename separated by '_'
                # Extract numeric parts using regular expressions
                match = re.search(r'\d+', frame_num_string)
                digits = match.group()
                frame_num = int(digits)
                
                # Group images by video name
                if video_name not in video_image_dict:
                    video_image_dict[video_name] = []
                video_image_dict[video_name].append({'img_file': image_path, 'frame_number': frame_num, "index":i, "video_name":video_name})
            else:
                video_image_dict[image_name] = [{'img_file': image_path, 'frame_number': 0, "index":i, "video_name":image_name}]
        
        # Now organize the total video information into self.videos
        for video_name, frames in video_image_dict.items():
            video_info = {
                'video_name': video_name,
                'frames': sorted(frames, key=lambda x: x['frame_number'])  # Sort frames by frame number
            }
            self.videos_info.append(video_info)
        
        return self.videos_info

    def split_sub_videos(self, interval, length, gpu_count, is_training=False):
        if not self.videos:
            assert self.length == 1, "For single image dataset, split_length should be 1"
            
        # Split video based on the specified interval_mode
        all_sub_videos = []

        self.length = length if is_training else -1
        # If validation mode, just output original videos
        if not is_training:
            for video_info in self.videos_info:
                frames = []
                for i, frame_info in enumerate(video_info['frames']):
                    frame_info["sub_frame_number"] = i
                    frames.append(frame_info)
                all_sub_videos.append(frames)    
            # all_sub_videos = all_sub_videos[:8]
            
            self.sub_videos = all_sub_videos
            sampler_indices, index_mapping_frameinfo = self._precompute_index_mapping(all_sub_videos, is_training)
            return self.sub_videos, sampler_indices, index_mapping_frameinfo

        if length == -1:
            length = min([len(video_info['frames']) for video_info in self.videos_info])
            self.length = length
        
        # Sampling sub-videos based on interval_mode
        for video_info in self.videos_info:
            frames = video_info['frames']
            sampled_frames = []

            if self.interval_mode == 'all':
                # All frames sampled in interval steps
                for i in range(0, len(frames), interval):
                    sampled_frames.append(frames[i])

                # Further splitting each video into sub-videos of length self.length
                sub_videos = [sampled_frames[i:i + length] for i in range(0, len(sampled_frames), length)]
            elif self.interval_mode == 'interval':
                # Interval sampling without all frames
                sampled_frames = frames[::interval]

                # Further splitting each video into sub-videos of length self.length
                sub_videos = [sampled_frames[i:i + length] for i in range(0, len(sampled_frames), length)]
            # Prepare sub-videos with img_file, frame_number, sub_frame_number
            for sub_video in sub_videos:
                sub_video_info = []
                if len(sub_video) != length:
                    continue
                for sub_frame_number, frame_info in enumerate(sub_video):
                    frame_info['sub_frame_number'] = sub_frame_number
                    sub_video_info.append(frame_info)

                all_sub_videos.append(sub_video_info)
        
        # Discard the extra sub-videos if not divisible by GPU count and batch_size
        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)
        target_multiple = lcm(gpu_count, self.batch_size*self.batch_size)
        old_len = len(all_sub_videos)
        all_sub_videos = all_sub_videos[:(len(all_sub_videos) // target_multiple) * target_multiple]
        new_len = len(all_sub_videos)
        # all_sub_videos = all_sub_videos[:1000*gpu_count*self.batch_size*self.batch_size]
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
        if self.rank == 0 or self.rank == -1:
            print(f"\n*******************{'[Train]' if self.augment else '[Test]'}dataset split info************************")
            LOGGER.info(f"Found {len(self.videos_info)} orige videos")
            LOGGER.info(f"Split {len(all_sub_videos)} sub videos")
            LOGGER.info(f"len sub videos is {list(set([len(spi) for spi in all_sub_videos]))[:10]} (print 10 number)")
            LOGGER.info(f"len sub videos is change from {old_len} to {new_len} ")
            # print(f"muti_rank_indices_splits: ")
            # print(self.muti_rank_indices_splits)
            LOGGER.info(f"*************************************************")

        sampler_indices, index_mapping_frameinfo = self._precompute_index_mapping(all_sub_videos, is_training)
        # sampler_indices的内容索引all_sub_videos的视频位置
        return all_sub_videos, sampler_indices, index_mapping_frameinfo
            
    def _train_video(self, hyp, length):
        """Sets bbox loss and builds transformations."""
        # hyp.mosaic = 0.0
        self.transforms = self.build_transforms(hyp) 
        LOGGER.info(f"[batch_size]: {self.batch_size}, now train dataset convert to split_length: {length}   mode: split_length")
        self.sub_videos, self.sampler_indices, self.index_mapping_frameinfo = self.split_sub_videos(self.interval, length=length, 
                                                                        gpu_count = self.world_size, is_training = self.augment)

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)
        
    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x
    
    def img2label_paths(self,img_paths):
        # sa, sb = f'{self.images_dir}', f'{self.labels_dir}'  # /images/, /labels/ substrings
        return [path.replace(sa, sb).split('.')[0]+'.txt' for path, sa, sb  in zip(img_paths, self.images_dir, self.labels_dir)]
    
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = self.img2label_paths(self.im_files)
        # cache_path = Path(combine_unique_folders([os.path.splitext(p)[0] for p in self.img_path])).with_suffix('.cache')
        cache_path = Path(self.im_files[0]).with_suffix(".cache") if isinstance(self.img_path, list) else Path(self.img_path).with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels
        
    def _set_samevideo_transform(self, seed):
        # Get the current time as a random number seed
        # seed = int(time.time())
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max([len(s) for s in segments])
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label
    
    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.index_mapping_frameinfo)

    
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        frame_info = self.index_mapping_frameinfo[index]
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        
        
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation

        label["img_metas"] = dict(frame_info)
        label["video_name"] = frame_info["video_name"]

        label["img_metas"]["is_first"] = frame_info["sub_frame_number"]==0
        
        if self.rect:
            label['rect_shape'] = np.ceil(np.array(label['resized_shape']) / self.stride + 0.5).astype(int) * self.stride
        return self.update_labels_info(label)    

    def plot_used(self, rank, cnt_use):
        """Plot the distribution of used frames."""
        import matplotlib.pyplot as plt
        # 获取使用次数列表
        usage_counts = list(cnt_use.values())
        # 获取元素标识列表
        elements = list(cnt_use.keys())
        # 绘制柱状图
        plt.bar(elements, usage_counts)
        plt.title("Distribution of used frames")
        plt.xlabel("Number of times used")
        plt.ylabel("Number of frames")
        path = f"/data/shuzhengwang/project/ultralytics/results/used_frames_{rank}.png"
        plt.savefig(path)
        print(f"Saved used frames plot to {path}")
    
    def _get_video_frame_indices(self):
        # Generate indices for each video, where each index represents a frame in the video
        video_frame_indices = []
        for video_idx, video in enumerate(self.sub_videos):
            video_frame_indices.append([video_idx] * len(video))  # Each video has a list of indices for its frames
        return video_frame_indices

    def _precompute_index_mapping(self, all_sub_videos, is_training):
        """
        Precompute the mapping from global index to (video_idx, frame_idx).
        
        Returns:
            dict: A dictionary mapping global frame index to (video_idx, frame_idx).
        """
        world_size = self.world_size if is_training else 1
        assert len(all_sub_videos)%world_size == 0, "Number of videos should be divisible by number of GPUs"

        index_mapping_frameinfo = {}
        for video_idx, video in enumerate(all_sub_videos):
            for frame_idx, frame in enumerate(video):
                index_mapping_frameinfo[frame["index"]] = frame
                
        indices = list(range(len(all_sub_videos)))
        # 将索引分成 self.world_size 份数的列表
        chunk_size = len(indices) // world_size
        index_chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        
        first_frame_per_GPU = [all_sub_videos[inds[0]] for inds in index_chunks]
        first_frame_per_GPU = [v[:4]  for v in first_frame_per_GPU]
        
        LOGGER.info(f"First_frame_per_GPU (video_idx, frame_number):\n")
        for gpu_video in first_frame_per_GPU:
            for f in gpu_video:
                LOGGER.info(f"{f}")
                
        assert any([f[0]["sub_frame_number"]==0 for f in first_frame_per_GPU]), "The first frame of each video should be in the same GPU"

        return index_chunks, index_mapping_frameinfo
    
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        video_list = self.sub_videos[index]
        video_trans_dict = []
        for i, frame in enumerate(video_list):
            orige_dict = self.get_image_and_label(frame["index"])
            self._set_samevideo_transform(index+self.epoch*10) #Same video in one epoch with consistent random seeds
            trans_dict = self.transforms(orige_dict.copy())
            trans_dict["index"] = index
            trans_dict["img_metas"] = orige_dict["img_metas"]
            
            video_trans_dict.append(trans_dict)
        return video_trans_dict  

        # if orige_dict["img_metas"]["is_first"]:
        #     support_trans_dict = trans_dict.copy()
        # else:
        #     video_idx_last, frame_idx_last, _ = self.get_video_frame_from_index(index-1, self.sub_videos_mapping)
        #     assert video_idx_last == video_idx and frame_idx_last == frame_idx-1, "The previous frame should be in the same video"
        #     support_dict = self.get_image_and_label(index-1)
        #     support_trans_dict = self.transforms(support_dict.copy())
        # trans_dict['img_ref'] = self.get_ref_img(orige_dict["neg_idx"][0]) #The most recent frame
        # motion = self._homoDta_preprocess(tensor_numpy(trans_dict["img"]),tensor_numpy(trans_dict['img_ref']))
        # trans_dict.update(motion)
        
    @staticmethod
    def collate_fn(batch_videos):
        """Collates data samples into batches."""
        length = len(batch_videos[0])
        new_batch_videos = []
        
        for f in range(length):
            batch = [batch_videos[i][f] for i in range(len(batch_videos))]
            new_batch = {}
            keys = batch[0].keys()
            values = list(zip(*[list(b.values()) for b in batch]))
            for i, k in enumerate(keys):
                value = values[i]
                if k == "img":
                    value = torch.stack(value, 0)
                if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", 'support_bboxes'}:
                    value = torch.cat(value, 0)
                new_batch[k] = value
            new_batch["batch_idx"] = list(new_batch["batch_idx"])
            for i in range(len(new_batch["batch_idx"])):
                new_batch["batch_idx"][i] += i  # add target image index for build_targets()
            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
            
            new_batch_videos.append(new_batch)
        return new_batch_videos
    
class YOLOVideoONXXDataset(YOLOVideoDataset):
    """

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        orige_dict = self.get_image_and_label(index)
        self._set_samevideo_transform(orige_dict["seed"]+self.epoch*10) #Same video in one epoch with consistent random seeds
        trans_dict = self.transforms(orige_dict.copy())
        
        if 'pos_id' in orige_dict:
            support_dict = self.get_image_and_label(orige_dict['pos_id'])
            support_trans_dict = self.transforms(support_dict.copy())
        else:
            support_trans_dict = trans_dict.copy()
        # trans_dict['img_ref'] = self.get_ref_img(orige_dict["neg_idx"][0]) #The most recent frame
        # motion = self._homoDta_preprocess(tensor_numpy(trans_dict["img"]),tensor_numpy(trans_dict['img_ref']))
        # trans_dict.update(motion)

        
        # self.show_transforms(orige_dict,trans_dict)
        trans_dict["index"] = index
        trans_dict["img_metas"] = orige_dict["img_metas"]
        trans_dict["support_bboxes"] = support_trans_dict["bboxes"]
        return trans_dict  
    
    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", 'support_bboxes'}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

from typing import List, Dict, Any, Optional
class YOLOStreamDataset(YOLODataset):
    """
    用于加载视频数据的YOLO数据集类。
    视频被视为帧的序列，每个视频的帧存储在单独的目录中。
    此类将视频划分为子视频进行处理。

    Args:
        video_length (int): 每个子视频的帧数 (训练时)。
        video_interval (int): 采样两个连续子视频之间的帧间隔 (训练时)。
        *args, **kwargs: 传递给父类 YOLODataset 的参数。
    """
    def __init__(self, data=None, *args, **kwargs):
        # 从 kwargs 中提取 'augment' 以决定采样策略
        # 父类 __init__ 中会设置 self.augment
        self.is_training_augment = kwargs.get('augment', True) # 默认训练时增强
        self.data = data
        if "interval" in data and kwargs["augment"]:
            video_interval = data["interval"]
            del data["interval"] # 删除以避免冲突
        else:
            video_interval = 1
        # 在调用 super().__init__ 之前处理 interval，确保它不会干扰视频帧的初始加载
        # 如果父类 YOLODataset 的 'interval' 参数与视频采样逻辑冲突，
        # 我们可能需要在这里显式地将其设置为1，让父类加载所有可能的帧。
        # 例如: parent_interval = kwargs.pop('interval', 1)
        # super().__init__(*args, interval=parent_interval, **kwargs)
        # 然而，根据您提供的BaseDataset代码，interval是其__init__的一部分。
        # YOLODataset会把data.yaml中的interval传递给BaseDataset。
        # 这里我们假设父类的interval处理的是全局帧采样，而video_length/video_interval是更细致的视频内采样。
        
        super().__init__(*args,data=data, **kwargs) # 调用 YOLODataset 的初始化

        self.video_length = video_length
        self.video_interval = video_interval
        
        # self.im_files 和 self.labels 已经由父类填充
        # self.ni 是总图像（帧）数

        self.sub_videos: List[List[int]] = []  # 存储每个子视频的帧索引(相对于原始self.im_files)
        self._organize_videos_and_subsample()

        # 如果采样后没有子视频，则发出警告
        if not self.sub_videos:
            LOGGER.warning(f"{self.prefix}No sub-videos were created. Check dataset structure, video_length, and video_interval.")
        else:
            # 计算所有子视频中的总有效帧数，这对于某些 sampler 可能有用
            self.total_sub_video_frames = sum(len(sv) for sv in self.sub_videos)
            LOGGER.info(
                f"{self.prefix}Created {len(self.sub_videos)} sub-videos with a total of "
                f"{self.total_sub_video_frames} frames to be sampled."
            )

    def _organize_videos_and_subsample(self):
        """
        识别原始视频，并根据 video_length 和 video_interval 将它们划分为子视频。
        子视频信息存储在 self.sub_videos 中，其中每个元素是原始 self.im_files 中的帧索引列表。
        """
        LOGGER.info(f"{self.prefix}Organizing videos and creating sub-videos...")
        original_videos: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)

        # 1. 根据目录对帧进行分组，以识别原始视频
        #    并记录它们在原始 self.im_files 中的索引
        for i, img_file_path in enumerate(self.im_files):
            video_dir = Path(img_file_path).parent
            # 帧名应能排序，如 "frame_0000001.jpg"
            frame_name = Path(img_file_path).name
            original_videos[video_dir].append({"original_idx": i, "path": img_file_path, "name": frame_name})

        # 2. 对每个原始视频的帧按名称排序，然后创建子视频
        for video_dir, frames_info in original_videos.items():
            # 按帧名排序，确保帧的顺序正确
            # 假设帧名格式为 'frame_xxxxxxx.jpg' 或类似，可以直接按字符串排序
            frames_info.sort(key=lambda x: x["name"])
            
            original_frame_indices = [f_info["original_idx"] for f_info in frames_info][::self.video_interval]
            num_frames_in_original_video = len(original_frame_indices)

            if num_frames_in_original_video == 0:
                continue

            if self.is_training_augment: # 训练模式：应用 video_length 和 video_interval
                if num_frames_in_original_video < self.video_length:
                    # 如果原始视频太短，可以跳过，或者将其作为一个子视频（如果长度大于0）
                    # pass
                    LOGGER.warning(
                        f"{self.prefix}Video {video_dir} has {num_frames_in_original_video} frames, "
                        f"less than video_length={self.video_length}. "
                    )
                else:
                    for i in range(0, num_frames_in_original_video - self.video_length + 1, self.video_length):
                        sub_video_indices = original_frame_indices[i : i + self.video_length]
                        self.sub_videos.append(sub_video_indices)
            else: # 测试/验证模式：将整个视频作为一个子视频
                self.sub_videos.append(original_frame_indices[:])
        
        if not self.sub_videos and original_videos: # 有视频文件但没生成子视频
             LOGGER.warning(f"{self.prefix}Found video data but no sub-videos were generated. "
                            f"This might be due to all videos being shorter than video_length in training mode.")
        elif not original_videos:
            LOGGER.warning(f"{self.prefix}No video directories found or parsed from im_files.")

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))
    # __getitem__ 方法会从父类 YOLODataset 继承。
    # 它期望的 `index` 是指向原始 `self.im_files` 和 `self.labels` 的索引。
    # Sampler 将负责提供这些正确的索引。

    # __len__ 方法也从父类继承，返回的是 `len(self.labels)`，即总帧数。
    # 这对于 DistributedVideoSampler 的逻辑是足够的，因为它会基于 `self.sub_videos` 构建自己的索引。
    # 或者，我们可以让 __len__ 反映子视频的数量，但这取决于 Sampler 如何设计。
    # 为了与 YOLODataset 的其他部分（如缓存）兼容，最好保持 __len__ 为总帧数。
    # Sampler 将不直接使用 dataset 的 __len__ 来确定迭代次数。

        
class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms


class GroundingDataset(YOLODataset):
    """Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format."""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes a GroundingDataset for object detection, loading annotations from a specified JSON file."""
        assert task == "detect", "`GroundingDataset` only support `detect` task for now!"
        self.json_file = json_file
        super().__init__(*args, task=task, data={}, **kwargs)

    def get_img_files(self, img_path):
        """The image files would be read in `get_labels` function, return empty list here."""
        return []

    def get_labels(self):
        """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image."""
        labels = []
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f'{x["id"]:d}': x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                cat_name = " ".join([img["caption"][t[0] : t[1]] for t in ann["tokens_positive"]])
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        return labels

    def build_transforms(self, hyp=None):
        """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))
        return transforms


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return YOLODataset.collate_fn(batch)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "WARNING ⚠️ Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
