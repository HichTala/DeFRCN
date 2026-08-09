import copy
import json
import logging
import os
import tempfile
from collections import OrderedDict

import numpy as np
import torch
import wandb
from fsdetection import load_fs_dataset

from fvcore.nn.precise_bn import get_bn_modules

from detectron2.utils import comm
from detectron2.engine import launch, HookBase, hooks
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import BoxMode
from defrcn.config import get_cfg, set_global_cfg
from defrcn.dataloader import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from defrcn.evaluation import DatasetEvaluators, verify_results, DatasetEvaluator, inference_on_dataset, \
    print_csv_format
from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup, EvalHookDeFRCN


class WandbHook(HookBase):
    def __init__(self, log_period=20):
        self.log_period = log_period

    def after_step(self):
        if self.trainer.iter % self.log_period == 0:
            metrics = self.trainer.storage.latest()
            log_dict = {
                k: v[0] for k, v in metrics.items()
                if isinstance(v, tuple)
            }
            log_dict["iter"] = self.trainer.iter
            wandb.log(log_dict)

    def after_train(self):
        metrics = self.trainer.storage.latest()
        log_dict = {
            f"test_{k}": v[0] for k, v in metrics.items()
            if isinstance(v, tuple)
        }
        log_dict["iter"] = self.trainer.iter
        wandb.log(log_dict)

class DatasetMapperHuggingFace(DatasetMapper):
    def __init__(self, cfg, is_train=True, is_validation=False, hf_dataset=None):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.is_validation = is_validation

        self.hf_dataset = hf_dataset
        if is_validation:
            self.image_dict = DatasetCatalog.get(cfg.DATASETS.VAL[0] + "_images")
        else:
            self.image_dict = DatasetCatalog.get(cfg.DATASETS.TEST[0] + "_images")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        if self.is_train:
            sample = self.hf_dataset[dataset_dict["image_id"]]
            image = sample["image"]
        else:
            image = self.image_dict[dataset_dict["image_id"]]

        conversion_format = self.img_format
        if self.img_format == "BGR":
            conversion_format = "RGB"

        image = image.convert(conversion_format)
        image = np.asarray(image)

        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

def hf_to_detectron2(dataset, split="train"):
    records = []

    for idx, sample in enumerate(dataset):
        width, height = sample["image"].size

        record = {
            "file_name": None,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": [],
        }

        for bbox, cat_id in zip(
                sample["objects"]["bbox"],
                sample["objects"]["category"]
        ):
            record["annotations"].append({
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id,
            })

        # record["image"] = sample["image"]
        records.append(record)

    return records

def hf_to_coco_dict(dataset, categories):
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    images_dict = {}

    ann_id = 1

    for img_id, sample in enumerate(dataset):
        width, height = sample["image"].size

        coco["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": f"{img_id}.jpg",
        })
        images_dict[img_id] = sample["image"]

        for bbox, cat_id in zip(
            sample["objects"]["bbox"],
            sample["objects"]["category"]
        ):
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            ann_id += 1

    return coco, images_dict

def write_temp_coco(coco_dict):
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode='w', delete=False
    )
    json.dump(coco_dict, tmp)
    tmp.close()
    return tmp.name

def register_hf_data():
    seed = os.getenv("REPEAT_ID", 2026)
    dataset_name = os.getenv("DATASET")

    dataset = load_fs_dataset(f"/lustre/fsn1/projects/rech/mvq/ubc18yy/datasets/{dataset_name}")
    og_dataset = copy.deepcopy(dataset["train"])
    classes = dataset["train"].features["objects"]["category"].feature.names

    id2label = dict(enumerate(classes))
    categories = [{"id": i, "name": name} for i, name in id2label.items()]

    coco_dict, images_dict_test = hf_to_coco_dict(dataset["test"], categories=categories)
    coco_path = write_temp_coco(coco_dict)

    register_coco_instances(f"{dataset_name}_test", {}, coco_path, image_root=".")
    DatasetCatalog.register(f"{dataset_name}_test_images", lambda: images_dict_test)
    MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=classes, evaluator_type="coco")
    del coco_dict

    coco_dict, images_dict_val = hf_to_coco_dict(dataset["validation"], categories=categories)
    coco_path = write_temp_coco(coco_dict)

    register_coco_instances(f"{dataset_name}_val", {}, coco_path, image_root=".")
    DatasetCatalog.register(f"{dataset_name}_val_images", lambda: images_dict_val)
    MetadataCatalog.get(f"{dataset_name}_val").set(thing_classes=classes, evaluator_type="coco")
    del coco_dict

    name = f"{dataset_name}_train"
    records = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

    name = f"{dataset_name}_1shot"
    dataset["train"].sampling(shots=1, seed=int(seed))
    records_1shot = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records_1shot)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

    name = f"{dataset_name}_5shot"
    dataset["train"].sampling(shots=5, seed=int(seed))
    records_5shot = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records_5shot)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

    name = f"{dataset_name}_10shot"
    dataset["train"].sampling(shots=10, seed=int(seed))
    records_10shot = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records_10shot)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

    del dataset
    return og_dataset

class Trainer(DefaultTrainer):

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = (
            0  # save some memory and time for PreciseBN
        )

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            repeat_id = os.getenv("REPEAT_ID", 2026)
            project = os.getenv("PROJECT", "DeFRCN")
            wandb.init(
                project=project,
                name=f"{cfg.DATASETS.TRAIN[0]}_{cfg.SOLVER.MAX_ITER}_rep{repeat_id}",
                group=f"{cfg.DATASETS.TRAIN[0]}_{cfg.SOLVER.MAX_ITER}",
                config=cfg
            )
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
            ret.append(WandbHook(log_period=20))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(EvalHookDeFRCN(
            cfg.TEST.EVAL_PERIOD, test_and_save_results, self.cfg))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            from defrcn.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(dataset_name, True, output_folder))
        if evaluator_type == "pascal_voc":
            from defrcn.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`defrcn.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        dataset = register_hf_data()
        mapper = DatasetMapperHuggingFace(cfg, is_train=True, hf_dataset=dataset)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, is_validation=True):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = DatasetMapperHuggingFace(cfg, is_train=False, is_validation=is_validation)
        return build_detection_test_loader(cfg, dataset_name, mapper)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(
                evaluators
            ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))
            if "VAL" in cfg.DATASETS:
                assert len(cfg.DATASETS.VAL) == len(evaluators), "{} != {}".format(
                    len(cfg.DATASETS.VAL), len(evaluators)
                )
            else:
                is_validation = False

        if is_validation:
            dataset_names = cfg.DATASETS.VAL
        else:
            dataset_names = cfg.DATASETS.TEST

        results = OrderedDict()
        for idx, dataset_name in enumerate(dataset_names):
            data_loader = cls.build_test_loader(cfg, dataset_name, is_validation)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, cfg)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        dataset_name
                    )
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    cfg = get_cfg()
    cfg.DATASETS.VAL = ()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
