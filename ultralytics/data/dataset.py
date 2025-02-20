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
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

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
        self.im_frame_matching(self.im_files)
        self.epoch = 0
        self.data_fre = [0] * len(self.img_video_info)
    
    def from_coco_get_image_id(self,file_name_mapping_id,im_file):
        if file_name_mapping_id:
            return file_name_mapping_id.get(im_file, 0)
        return 0
    
    def video_sampler_split(self, video_image_dict, mode="all",length=100, raandom_seed=100):
        '''
        mode: all split_random split_legnth
        interval_mode all[[1,3,5],[2,4,6]], one[[1,3,5]],
        '''
        
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
        
        
        interval_mode = self.data["interval_mode"] if "interval_mode" in self.data else "all"
        random.seed(raandom_seed)
        if mode=="all":
            self.length = -1
            self.sub_video_splits = []
            min_length = 0
            for video_name,video_list in video_image_dict.items():
                min_length = min(min_length, len(video_list))

            for video_name,video_list in video_image_dict.items():
                self.sub_video_splits.append(video_list)
                    
        elif mode=="split_legnth":
            min_video_lengthes = min([len(video_list)-1 for video_list in video_image_dict.values()])
            self.length = min(min_video_lengthes,length) #Video segments should not be too long
            print(f"min length video is {self.length}")
            self.sub_video_splits = []
            # Get the full division of self.interval interval for each video
            if interval_mode=="all":
                for video_name,video_list in video_image_dict.items():
                    for i in range(self.interval):
                        sub_interval_list = video_list[i::self.interval]
                        sub_interval_length_list = [sub_interval_list[j:j + self.length] for j in range(0, len(sub_interval_list), self.length)]
                        sub_interval_length_list[-1] = sub_interval_list[-self.length:]
                        for sub_ in sub_interval_length_list:
                            if len(sub_) == self.length:
                                self.sub_video_splits.append(sub_)
            else:
                for video_name,video_list in video_image_dict.items():
                    sub_interval_list = video_list[::self.interval]
                    sub_interval_length_list = [sub_interval_list[j:j + self.length] for j in range(0, len(sub_interval_list), self.length)]
                    sub_interval_length_list[-1] = sub_interval_list[-self.length:]
                    for sub_ in sub_interval_length_list:
                        if len(sub_) == self.length:
                            self.sub_video_splits.append(sub_)
            # import pdb;pdb.set_trace()
            if world_size > 1: #muti gpu
                world_size = dist.get_world_size()
                if len(self.sub_video_splits) % world_size != 0:#Can't average the scores.
                    len_sub_video_splits = len(self.sub_video_splits)
                    nearest_multiple_of_3 = ((len_sub_video_splits - 1) // world_size + 1) * world_size
                    difference = nearest_multiple_of_3 - len_sub_video_splits
                    for i in range(difference):
                        self.sub_video_splits.append(list(self.sub_video_splits[i]))
        elif mode=="split_random":
            print(f"min length rate video is {length}")
            self.sub_video_splits = []
            # Get the full division of self.interval interval for each video
            for video_name,video_list in video_image_dict.items():
                video_length = len(video_list)-1
                max_rate = video_length//min(length, video_length) + 1
                split_length = random.choice([video_length//rate for rate in range(1,max_rate)])

                if interval_mode=="all":
                    for i in range(self.interval):
                        sub_interval_list = video_list[i::self.interval]
                        sub_interval_length_list = [sub_interval_list[j:j + split_length] for j in range(0, len(sub_interval_list), split_length)]
                        sub_interval_length_list[-1] = sub_interval_list[-split_length:]
                        for sub_ in sub_interval_length_list:
                            # if len(sub_) == split_length:
                            self.sub_video_splits.append(sub_)
                else:
                    sub_interval_list = video_list[::self.interval]
                    sub_interval_length_list = [sub_interval_list[j:j + split_length] for j in range(0, len(sub_interval_list), split_length)]
                    sub_interval_length_list[-1] = sub_interval_list[-split_length:]
                    for sub_ in sub_interval_length_list:
                        self.sub_video_splits.append(sub_)

        # self.sub_video_splits = self.sub_video_splits[:6] #debug
        indices = list(range(len(self.sub_video_splits)))
        random.shuffle(indices)
        self.muti_rank_indices_splits, self.muti_rank_sub_video_len, self.per_gpu_total_frames = self.split_video_frames(self.sub_video_splits, indices, world_size)
        
        if rank == 0 or rank == -1:
            print(f"\n*******************{'[Train]' if self.augment else '[Test]'}dataset split info************************")
            print(f"len sub videos is {list(set([len(spi) for spi in self.sub_video_splits]))[:10]} (print 10 number)")
            print(f"per GPU frames len: {self.muti_rank_sub_video_len}")
            print(f"per GPU video number: {[len(sub_indexs) for sub_indexs in self.muti_rank_indices_splits]}")
            # print(f"muti_rank_indices_splits: ")
            # print(self.muti_rank_indices_splits)
            print(f"*************************************************")

        #init the first frame of the subvideo
        for info in self.img_video_info:
            if "seed" in info:
                del info["seed"]
            if "is_first" in info:
                info["is_first"] = []
        i = 0
        for gpu_video_index_list in self.muti_rank_indices_splits: #Storing a different seed for each video that the gpu may access means that if there are videos or images that are accessed twice with different seed
            # At the same time dataset get item will take seed in order and use the
            for index_video in gpu_video_index_list:
                sub_list = self.sub_video_splits[index_video]
                for index, frame in enumerate(sub_list):
                    if index==0:
                        self.img_video_info[frame["index"]]["is_first"].append(True)
                    else:
                        self.img_video_info[frame["index"]]["is_first"].append(False)


                    if "seed" in self.img_video_info[frame["index"]]:
                        self.img_video_info[frame["index"]]["seed"].append(i*5) #Belongs to muti videos
                    else:
                        self.img_video_info[frame["index"]]["seed"] = [i*5]  #Video Enhanced Random Seeds, One Video One Seed
                i += 1
                    

    def video_init_split(self, video_image_dict):
        if self.augment:
            self.video_sampler_split(video_image_dict, mode="split_random", length=8)
        else:
            self.video_sampler_split(video_image_dict, mode="all")
            

    def end_train_all_video(self):
        print(f"change data video split closed")
        self.video_sampler_split(self.video_image_dict.copy(), mode="all")

    def split_video_frames(self, sub_videos_list, indices, n):
        '''
        The multi-card environment allocates the training data on each GPU, and returns a list of length n. Each list represents the index of a part of the sub_videos_list, and the total length of the n sub-lists is similar.
        partitions_len is the length of each partition.
        return the shortest total number of frames as the total length of training min_total_frames, progress bar display: min_total_frames//batch_size
        '''
        # Calculate the total number of video frames for each sub-list
        total_frames = [len(video) for video in sub_videos_list]
        # print(f"now dataset total frame is: {total_frames}")
        # Calculate the total number of video frames for all sublists
        total_frames_sum = sum(total_frames)
        
        # Calculate the total number of video frames each sublist should contain
        target_frames_per_partition = total_frames_sum // n
        
        # Initialization Result List
        partitions = [[] for _ in range(n)]
        partitions_len = [0 for _ in range(n)]

        current_partition = 0
        current_partition_frames = 0
        
        # Iterate through each sub-list
        for video_index in indices:
            frames_count = total_frames[video_index]
            # If the number of video frames in the current partition has exceeded the target, move to the next partition
            if current_partition_frames + frames_count > target_frames_per_partition:
                partitions[current_partition].append(video_index)
                partitions_len[current_partition] += frames_count

                current_partition += 1
                current_partition_frames = 0
                continue
            
            # Add the current sublist to the current partition
            partitions[current_partition].append(video_index)
            partitions_len[current_partition] += frames_count
            current_partition_frames += frames_count
        
        if partitions_len[current_partition] < target_frames_per_partition:
            video_index = indices[0]
            partitions[current_partition].append(video_index)
            partitions_len[current_partition] += frames_count

        return partitions, partitions_len, min(partitions_len)

    def get_index_from_sub(self, ix, iy): #ix indexes subvideo, iy indexes video frames
        return self.sub_video_splits[ix][iy]["index"]
            
        
    def im_frame_matching(self, im_files):
        # import json
        
        #val
        coco_data = None
        image_name_map_id = None
        if "eval_ann_json" in self.data:
            with open(self.data["eval_ann_json"], 'r', encoding='utf-8') as coco_file:
                coco_data = json.load(coco_file)
            image_name_map_id = {}
            for image in coco_data["images"]:
                image_name_map_id[image["file_name"]] = image["id"]

        # Create a dictionary that groups images by video name
        video_image_dict = {}
        img_video_info = []
        for i,image_path in enumerate(im_files):
            video_name = os.path.basename(os.path.dirname(image_path))
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            frame_num_string = image_name.split('_')[-1] # Assuming the video name is the first part of the filename separated by '_'
            # Extract numeric parts using regular expressions
            match = re.search(r'\d+', frame_num_string)
            digits = match.group()
            frame_num = int(digits)
            # print(f"string: {frame_num_string} -> {frame_num}")
            
            if video_name not in video_image_dict:
                video_image_dict[video_name] = []
            video_image_dict[video_name].append({
                "index":i,
                "frame_number":frame_num
            })
            img_video_info.append({
                "frame_number":frame_num,
                "video_name":video_name,
                "image_id":self.from_coco_get_image_id(image_name_map_id,video_name+"/"+os.path.basename(image_path))
            })

        # Sort each video by frame number
        for key,value in video_image_dict.items():
            sorted_value = sorted(value, key=lambda x: x['frame_number'])
            video_image_dict[key] = sorted_value
        
    
        # Frame information before and after writing
        for video_name,video_list in video_image_dict.items(): 
            video_first_frame = video_list[0]["frame_number"] + self.match_number * self.interval
            video_last_frame = video_list[-1]["frame_number"] - self.match_number * self.interval
            # Generate indexes in the negative direction (excluding 0)
            neg_idxs = np.arange(-self.interval, -self.interval * self.match_number - 1, -self.interval)
            # Generate indexes in the positive direction (excluding 0)
            pos_idxs = np.arange(self.interval, self.match_number * self.interval + 1, self.interval)

            for i,frame in enumerate(video_list):
                # import pdb;pdb.set_trace()
                index = frame["index"]
                cur_frame_number = frame["frame_number"]
                # Starting and ending frame numbers for training
                img_video_info[index]["video_first_frame"] = video_first_frame
                img_video_info[index]["video_last_frame"] = video_last_frame
                neg_idx_cur = (neg_idxs + i).clip(0)
                pos_idx_cur = (pos_idxs + i).clip(0,len(video_list)-1)

                neg_idx_cur = neg_idx_cur if cur_frame_number  >= video_first_frame else None
                pos_idx_cur = pos_idx_cur if cur_frame_number <= video_last_frame else None

                #It's all there in order to train.
                if (neg_idx_cur is not None) and (pos_idx_cur is not None):
                    img_video_info[index]["is_train"] = True
                    img_video_info[index]["neg_idx"] = [video_list[idx]["index"] for idx in neg_idx_cur]
                    img_video_info[index]["pos_idx"] = [video_list[idx]["index"] for idx in pos_idx_cur]
                else:
                    img_video_info[index]["is_train"] = False
                
                img_video_info[index]["is_first"] = (cur_frame_number == video_first_frame)

        self.img_video_info = img_video_info
        self.video_image_dict = video_image_dict
        self.video_init_split(video_image_dict.copy())
        
        return True

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
        
    def _set_samevideo_transform(self, seed):
        # Get the current time as a random number seed
        # seed = int(time.time())
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def _train_video(self, hyp, index):
        """Sets bbox loss and builds transformations."""
        # hyp.mosaic = 0.0
        self.transforms = self.build_transforms(hyp) 
        LOGGER.info(f"now train dataset convert to split_length: {self.data['split_length'][index]}   mode: split_random")
        self.video_sampler_split(self.video_image_dict.copy(), mode="split_random", length=self.data["split_length"][index])

    def _train_backbone(self, hyp):
        """Sets bbox loss and builds transformations."""
        # hyp.mosaic = 1.0
        self.transforms = self.build_transforms(hyp) 
        self.video_sampler_split(self.video_image_dict.copy(), mode="split_random", length=self.data["split_length"][0])

    def _train_all(self, hyp):
        """Sets bbox loss and builds transformations."""
        # hyp.mosaic = 1.0
        self.transforms = self.build_transforms(hyp) 
        self.video_sampler_split(self.video_image_dict.copy(), mode="all", length=self.data["split_length"][0])
    
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
        
    # Take the current seed that should be used (as opposed to selecting the video that the current image belongs to)
    def select_now_seed(self, seed_list):
        seed = seed_list[0]
        del seed_list[0]
        seed_list.append(seed)
        return seed, seed_list

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
    
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        image_info = self.img_video_info[index]
        is_train = image_info["is_train"]
        # if not train
        # while(not is_train):
        #     index = random.randint(0, len(self.img_video_info)-1)
        #     image_info = self.img_video_info[index]
        #     is_train = image_info["is_train"]
            
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        
        if image_info["is_train"]:
            label['pos_id'] = image_info["pos_idx"][0]
            label['neg_id'] = image_info["neg_idx"][0]
        # if is_train:
        #     label["pos_idx"],label["neg_idx"] = image_info["pos_idx"],image_info["neg_idx"]
        
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        label["image_id"] = image_info["image_id"]
        label["img_metas"] = {
            "frame_number":image_info["frame_number"],
            "video_name":image_info["video_name"],
            "epoch":self.epoch
        }

        if len(image_info["seed"]) > 1:#image belongs to multiple video clips
            label["seed"], image_info["seed"] = self.select_now_seed(image_info["seed"])
        else:
            label["seed"] = image_info["seed"][0]

        if len(image_info["is_first"]) > 1:#image belongs to multiple video clips
            label["img_metas"]["is_first"], image_info["is_first"] = self.select_now_seed(image_info["is_first"])
        else:
            label["img_metas"]["is_first"] = image_info["is_first"][0]

        # label["seed"] = self.epoch * 10
        # label["img_metas"]["is_first"] = False
        # LOGGER.info(f"Now is random train")
        
        if self.rect:
            label['rect_shape'] = np.ceil(np.array(label['resized_shape']) / self.stride + 0.5).astype(int) * self.stride
        return self.update_labels_info(label)    
          
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
    
    # @staticmethod
    # def collate_fn(batch):
    #     """Collates data samples into batches."""

    #     new_batch = {}
    #     keys = batch[0].keys()
    #     values = list(zip(*[list(b.values()) for b in batch]))
    #     for i, k in enumerate(keys):
    #         value = values[i]
    #         if k in ('img',"org_imgs","input_tensors","patch_indices","h4p"):
    #             value = torch.stack(value, 0)

    #         if k in ['masks', 'keypoints', 'bboxes', 'cls', 'support_bboxes']:
    #             value = torch.cat(value, 0)

    #         new_batch[k] = value
    #     new_batch['batch_idx'] = list(new_batch['batch_idx'])
    #     for i in range(len(new_batch['batch_idx'])):
    #         new_batch['batch_idx'][i] += i  # add target image index for build_targets()
    #     new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

    #     new_batch["img"] = {
    #         "backbone":new_batch["img"],
    #         "img_metas":new_batch["img_metas"]
    #     }
    #     # new_batch["cls"] = new_batch["is_moving"]
    #     return new_batch    
    
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
