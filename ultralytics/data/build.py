# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed
from ultralytics.utils import LOGGER
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOVideoDataset, YOLOMultiModalDataset, YOLOStreamDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    LoadTensorBuffer,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file

import torch
import random
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class StreamSampler(DistributedSampler):
    '''
    Streaming Sampling Datasets from Video
    '''
    def __init__(self, dataset, batch_size, seed=10,distributed=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.seed = seed
        if distributed:
            self.rank = dist.get_rank()  # Get the rank of the current process
            self.world_size = dist.get_world_size()  # Get the total number of processes
        else:
            self.rank = 0
            self.world_size = 1
            
        self.last_next_frame_index = [None for i in range(int(batch_size))]
        self.next_train_video_index = 0

        # random.seed(self.seed)
        # indices = list(range(len(self.dataset.sub_video_splits)))
        # random.shuffle(indices)
        
        # indices_per_rank = len(indices) // self.world_size  #  Data subset size per process
        # start_idx = self.rank * indices_per_rank
        # end_idx = (self.rank + 1) * indices_per_rank
        
        # self.indices = indices[start_idx:end_idx]  # Video subset of the current process
        # self.indices = self.dataset.muti_rank_indices_splits[self.rank]
        self.data_length = self.dataset.per_gpu_total_frames - self.dataset.per_gpu_total_frames%self.batch_size

        print(f"worksize:{self.world_size}, rank: {self.rank}")
    def __iter__(self):

        self.last_next_frame_index = [None for i in range(int(self.batch_size))]
        self.next_train_video_index = 0
        self.indices = self.dataset.muti_rank_indices_splits[self.rank]
        
        random.seed(44)
        random.shuffle(self.indices)
        
        self.data_length = self.dataset.per_gpu_total_frames - self.dataset.per_gpu_total_frames%self.batch_size
        # print(f"sampler init: {len(self.indices)}")

        i = 0
        for _ in range(self.data_length):
            if i == self.batch_size:
                i = 0
            
            if self.last_next_frame_index[i] is not None:
                ix,iy = self.last_next_frame_index[i] # Video indexing and frame indexing
                index = self.dataset.get_index_from_sub(ix,iy)
                cur_sampler_is_train = self.dataset.img_video_info[index]["is_train"]
            else: 
                cur_sampler_is_train = False
                
            if cur_sampler_is_train:                        #Current Video Continue Training
                yield index
                if iy + 1 == len(self.dataset.sub_video_splits[ix]):
                    self.last_next_frame_index[i] = None    #Video training complete.
                else:
                    self.last_next_frame_index[i][-1] += 1  #Continue training this video using the next frame
            else:                                           #Switching Video Training
                ix, iy = self.indices[self.next_train_video_index], -1
                self.next_train_video_index += 1
                if self.next_train_video_index == len(self.indices):
                    self.next_train_video_index = 0

                is_train = False
                # if not train, next
                while(not is_train):
                    iy += 1
                    index = self.dataset.get_index_from_sub(ix, iy)
                    is_train = self.dataset.img_video_info[index]["is_train"] if self.dataset.augment else True
                    if iy + 1 == len(self.dataset.sub_video_splits[ix]): 
                        self.last_next_frame_index[i] = None #Video training complete.
                    else:
                        self.last_next_frame_index[i] = [ix, iy+1] #Video Continues Training
                yield index
                
            i += 1
            
        self.last_next_frame_index = [None for i in self.last_next_frame_index]
        self.next_train_video_ix = 0
        
    def _epoch_reset(self):
        self.last_next_frame_index = [None for i in self.last_next_frame_index]
        self.next_train_video_ix = 0 

    def __len__(self):
        return self.data_length


class VideoSampler(DistributedSampler):
    '''
    Streaming Sampling Datasets from Video
    '''
    def __init__(self, dataset, batch_size, seed=10, distributed=False, shuffle=True):
        # Check if distributed mode is enabled and process group is already initialized
        self.dataset = dataset  # self.sub_videos
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.distributed = distributed
        self.shuffle = shuffle
        
        if distributed:
            self.rank = dist.get_rank()  # Get the rank of the current process
            self.world_size = dist.get_world_size()  # Get the total number of processes
        else:
            self.rank = 0
            self.world_size = 1
        
    def __iter__(self):
        # Generate indices for sub-videos (each sub-video is a list of frames)
        indices = list(self.dataset.sampler_indices[self.rank])
        
        # Shuffle if needed (for training)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
            if self.rank == 0 or self.rank == -1:
                print(f"epoch: {self.epoch}, shuffle indices: {indices[:5]}")
            
        for i in indices:
            yield i

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset.sampler_indices[self.rank])  # Number of batches

    def _epoch_reset(self):
        '''
        Reset for a new epoch: re-calculate the video frame indices and re-shuffle the dataset.
        This is necessary when the sub-video divisions change.
        '''
        # Recalculate the video frame indices if necessary (if you modified all_sub_videos)
        self.video_frame_indices = self.dataset._get_video_frame_indices()

        # Reset any other internal state as needed
        self.seed += 1  # Optionally increment the seed for the next epoch
        self.epoch += 1  # Increment the epoch
        self.set_epoch(self.epoch)  # Update the epoch in the distributed sampler

from torch.utils.data.sampler import BatchSampler
class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()

class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False,images_dir=None,
                 labels_dir=None,):
    """Build YOLO Dataset."""
    # dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    datasetname = data.get('datasetname', 'YOLODataset')
    if datasetname == "YOLODataset":
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    elif datasetname in ("YOLOStreamDataset",  "YOLOVideoDataset"):
        dataset = YOLOStreamDataset
    # elif datasetname == "YOLOVideoDataset":
    #     dataset = YOLOVideoDataset
    else:
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
        
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        images_dir = images_dir,
        labels_dir = labels_dir
    )
    

def build_yoloft_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False,images_dir=None,
                 labels_dir=None,):
    """Build YOLO Dataset."""
    # dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    datasetname = data.get('datasetname', 'YOLODataset')
    if datasetname == "YOLODataset":
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    elif datasetname in ("YOLOStreamDataset"):
        dataset = YOLOStreamDataset
    elif datasetname == "YOLOVideoDataset":
        dataset = YOLOVideoDataset
    else:
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
        
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        images_dir = images_dir,
        labels_dir = labels_dir
    )
    
def build_yoloft_val_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False,images_dir=None,
                 labels_dir=None,):
    """Build YOLO Dataset."""
    # dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    datasetname = data.get('datasetname', 'YOLODataset')
    if datasetname == "YOLODataset":
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    else:
        dataset = YOLOStreamDataset
        LOGGER.info("YOLOFT Network val and test use YOLOStreamDataset")
        
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        images_dir = images_dir,
        labels_dir = labels_dir
    )
    
def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )

def build_video_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = VideoSampler(dataset, batch, seed=1, distributed=False) if rank == -1 else VideoSampler(dataset, batch, seed=1, distributed=True, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )

def build_stream_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = StreamSampler(dataset, batch, seed=1, distributed=False) if rank == -1 else StreamSampler(dataset, batch, seed=1, distributed=True)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    
def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor, buffer = False, False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        if len(source)==2 and len(source[-1]) == 3:
            if isinstance(source[0], torch.Tensor):
                tensor = True
            else:
                from_img = True
            buffer = True
        else:
            source = autocast_list(source)  # convert all list elements to PIL or np arrays
            from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor, buffer


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor, buffer = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor, buffer)

    # Dataloader
    if tensor and buffer:
        dataset = LoadTensorBuffer(source[0], source[1])
    elif tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img and buffer:
        dataset = LoadPilAndNumpy(source[0], source[1])
        # AttributeError("error, img buffer have not achieve")
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
