import logging
import numpy as np
import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import torch

@DATASETS.register_module()
class NuScenesTDataset(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length=None,
        start_on_keyframe=True,
    ) -> None:
        self.video_length = 35 #video_length
        self.start_on_keyframe = start_on_keyframe
        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag)
        if "12Hz" in ann_file and start_on_keyframe:
            logging.warn("12Hz should use all starting frame to train, please"
                         "double-check!")

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for scene in scene_tokens:
            for start in range(len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                #if self.start_on_keyframe and len(scene[start] >= 33):
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.video_length]]
                all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length}-clip, "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        #print("32523452333333333333333333333333333333333",ann_file)
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        #print("32523452333333333333333333333333333333333",type(data_infos))
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens'])
        return data_infos

    def __len__(self):
        return len(self.clip_infos)

    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[index]
        frames = []
        for frame in clip:
            frame_info = super().get_data_info(frame)
            info = self.data_infos[frame]
            frames.append(frame_info)
        return frames

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
        """
        frames = self.get_data_info(index)
        
        
        
        
        
        if None in frames:
            return None
        examples = []
        for frame in frames:
        
            #print("Initial frame keys664546:", frame)
            #print("Initial frame key25646666666666666:",frame.keys())
            
            
            #print("Initial frame key25647777777777777777:",frame.keys())
            #print("888888888888889",frame["ann_info"]["gt_labels_3d"].shape)
            #print("888888888888880",frame["ann_info"]["instance_token_3d"].shape)
            gt_labels_3d = frame["ann_info"]["gt_labels_3d"]

            #
            gt_labels_3d[gt_labels_3d == -1] = 1
            
            #
            frame["ann_info"]["gt_labels_3d"] = gt_labels_3d
            
            
            
            #
            #frame["ann_info"]["instance_token_3d"] = instance_tokens_3d
            sate = frame["ann_info"]["instance_token_3d"]
            #print("Initial frame sate:", sate)
            #print("After pre_pipeline keys3526245:", frame.keys())
            self.pre_pipeline(frame)
            example = self.pipeline(frame)
            #print("66666666666666",example["ann_info"]["gt_labels_3d"].shape)
            #print("888888888888887",example)#["ann_info"]["instance_token_3d"].shape
            #print("sate",sate)
            #print(a)
            sate = np.vectorize(lambda x: x[:8], otypes=[str])(sate)
            #sate = np.array([int(hex_val, 16) for hex_val in sate])
            sate = np.array([[int(hex_val, 16)] for hex_val in sate])
            sate = sate.flatten()
            #print("sate0001",type(sate))
            
            
            condition = (sate >= 100000000) & (sate < 10000000000)
            
            #
            sate[condition] = sate[condition] * 10
            #digits = np.floor(np.log10(sate) + 1).astype(int)

            #
            #needed_powers = 10 - digits
            
            #
            #adjustment_factors = 10**np.where(digits < 11, needed_powers, 0)
            
            #
            #sate = sate * adjustment_factors
            
            
            sate = torch.from_numpy(sate)
            #print(a)
            example["instance_token_3d"]=sate
            #print("7777777777777777777777",sate)
            #print("888888888888888888888",frame["ann_info"]["gt_labels_3d"])
            #print("77777777777777777777778888888888",type(sate))
            #print("After pipeline keys12413513268888888888888888888888888888888888888886:", example.keys())
            
            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):  
                print("omggggggggggggggggggggg")
                return None
            
            
            #print("9999999999999",frame["ann_info"]["gt_labels_3d"].shape)
            #print("99999999999991",frame["ann_info"]["instance_token_3d"].shape)
            #print("99999999999992",frame["ann_info"]["gt_labels_3d"])
            
            #example["gt_labels_3d"]=frame["ann_info"]["gt_labels_3d"]
            #print("99999999999992",example["gt_labels_3d"].shape)
            #print("99999999999993",example["instance_token_3d"].shape)
            examples.append(example)
            #print("666666666666666666",examples)
        #print("12343",[example["gt_labels_3d"] for example in examples])
                
        '''
        print("gt_bboxes_3d",examples["gt_bboxes_3d"])
        print("gt_labels_3d",examples["gt_labels_3d"])
        print("instance_token_3d",examples["instance_token_3d"])
        print("examples keys:", examples.keys())
        print("oppppppppppppppppppppppppp")
        '''
        #print(c)
        '''
        print("gt_bboxes_3d",example["gt_bboxes_3d"].shape)
        print("gt_labels_3d",example["gt_labels_3d"].shape)
        print("instance_token_3d",example["instance_token_3d"].shape)
        
        for key in ["gt_bboxes_3d", "gt_labels_3d", "instance_token_3d"]:
            value = example.get(key)
        
            if value is not None and hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key} is either empty or does not have a shape attribute.")
        
        '''

        return examples
        
        
        
        
        
