from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import default_argument_parser, default_setup, launch

from dataset.dataset import get_hrb_dicts, get_frhrb_dicts
from configs.config import add_hrb_config
from utils.evaluator import HrbEvaluator
from utils.parameters import *


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return HrbEvaluator(dataset_name, cfg)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(MODEL_CONFIG)
    # cfg.merge_from_list(args.opts)
    add_hrb_config(cfg)

    if 'DATASETS.TRAIN' in args.opts:
        train_dataset = args.opts[args.opts.index('DATASETS.TRAIN') + 1]
        cfg.DATASETS.TRAIN = (train_dataset,)

    if 'DATASETS.TEST' in args.opts:
        test_dataset = args.opts[args.opts.index('DATASETS.TEST') + 1]
        cfg.DATASETS.TEST = (test_dataset,)

    if 'SOLVER.MAX_ITER' in args.opts:
        max_iter = int(args.opts[args.opts.index('SOLVER.MAX_ITER') + 1])
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.STEPS = ((max_iter * 2) // 3, (max_iter * 8) // 9)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    for d in ['all', 'train', 'test']:
        DatasetCatalog.register('hrb_paris_' + d, lambda d=d: get_hrb_dicts(d))
        MetadataCatalog.get('hrb_paris_' + d).set(
            thing_classes=ORGAN_LIST, split=d)
    for d in ['fr']:
        DatasetCatalog.register('hrb_' + d, lambda d=d: get_frhrb_dicts(d))
        MetadataCatalog.get('hrb_' + d).set(thing_classes=ORGAN_LIST)

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
