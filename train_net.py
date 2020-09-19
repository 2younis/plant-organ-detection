
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from dataset.dataset import get_hrb_dicts, get_frhrb_dicts
from configs.config import add_hrb_config, add_frhrb_config
from utils.evaluator import HrbEvaluator
from utils.parameters import *
import os


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return HrbEvaluator(dataset_name, cfg)  

    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(MODEL_CONFIG)
    #cfg.merge_from_list(args.opts)
    add_hrb_config(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    
    return cfg


def main(args):
    for d in ["train", "test"]:
        DatasetCatalog.register("hrb_" + d, lambda d=d: get_hrb_dicts(d))
        MetadataCatalog.get("hrb_" + d).set(thing_classes=['leaf', 'flower', 'fruit', 'seed', 'stem', 'root'],
            dirname="./", year= 2012, split=d)
    for d in ['fr']:
        DatasetCatalog.register(d + 'hrb', lambda d=d: get_frhrb_dicts(d))
        MetadataCatalog.get(d + 'hrb').set(thing_classes=['leaf', 'flower', 'fruit', 'seed', 'stem', 'root'])

    cfg = setup(args)


    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
