# segmentation_train.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset.gs_dataset import GaussianDataset
from utils.system_utils import set_seed
from model.mink_unet import mink_unet

#import MinkowskiEngine as ME
# 기본값 설정 (필요 시 config에서 오버라이드)
ELASTIC_DISTORT_PARAMS = {"granularity": 5, "magnitude": 0.2}
ROTATION_AXIS = [0, 1, 0]  # y축 기준 수평 반전

def collate_fn(batch):
    """Collate function for batching sparse tensor data."""
    locs, features, labels, head_id = list(zip(*batch))

    for i in range(len(locs)):
        locs[i][:, 0] *= i  # 배치 인덱스 추가

    return torch.cat(locs), torch.cat(features), torch.cat(labels), head_id[0]


def init_dir(config):
    """Initialize output directory and TensorBoard writer."""
    if not hasattr(config, 'exp_name') or not config.exp_name:
        config.exp_name = "semantic_seg_exp"
    output_dir = f"./results_semantic_seg/{config.exp_name}"

    print(f"Output folder: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as fp:
        OmegaConf.save(config, fp)

    os.makedirs(os.path.join(output_dir, "tb_logs"), exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, "tb_logs"))
    return writer, output_dir


def train(config):
    """Train MinkUNet for 3D semantic segmentation."""
    # 모델 초기화
    in_channels = 3 if config.input_type == "low" else config.model.backbone.in_channels
    model = mink_unet(
        in_channels=in_channels,
        out_channels=config.model.backbone.out_channels,
        D=3,
        arch=config.model.backbone.type.split("MinkUNet")[1].strip() if "MinkUNet" in config.model.backbone.type else "34C"
    ).cuda()

    writer, output_dir = init_dir(config)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=config.optimizer.nesterov
    )
    scheduler = torch.optim.OneCycleLR(
        optimizer,
        max_lr=config.scheduler.max_lr,
        total_steps=config.epochs * len(train_loader),  # 동적 계산 필요
        pct_start=config.scheduler.pct_start,
        anneal_strategy=config.scheduler.anneal_strategy,
        div_factor=config.scheduler.div_factor,
        final_div_factor=config.scheduler.final_div_factor
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.ignore_index)

    # 데이터셋 및 로더 설정
    train_dataset = GaussianDataset(
        gaussians_dir=config.data.train.gaussians_dir,
        input_type=config.input_type,
        gaussian_iterations=config.gaussian_iterations,
        voxel_size=config.voxel_size if config.use_voxelization else None,
        aug=config.aug,
        feature_type=config.feature_type,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 평가 데이터셋 및 로더
    val_dataset = GaussianDataset(
        gaussians_dir=config.data.val.gaussians_dir,
        input_type=config.input_type,
        gaussian_iterations=config.gaussian_iterations,
        voxel_size=config.voxel_size if config.use_voxelization else None,
        aug=False,
        feature_type=config.feature_type,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # AMP (Automatic Mixed Precision) 설정
    scaler = torch.cuda.amp.GradScaler(enabled=config.enable_amp)

    # 학습 루프
    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        range(config.epochs * len(train_loader)),
        desc="Training progress",
        dynamic_ncols=True,
    )

    for epoch in range(config.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            locs, features, labels, head_id = batch
            locs, features, labels = locs.cuda(), features.cuda(), labels.cuda()

            # Sparse Tensor 생성
            sinput = ME.SparseTensor(features=features, coordinates=locs, device="cuda")
            with torch.cuda.amp.autocast(enabled=config.enable_amp):
                output = model(sinput).F
                loss = criterion(output, labels)

            # Optimizer 및 스케줄러
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/loss", loss.item(), progress_bar.n)
            writer.add_scalar("lr", optimizer.state_dict()["param_groups"][0]["lr"], progress_bar.n)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(1)

        scheduler.step()

        # 평가
        if (epoch + 1) % config.test_interval == 0:
            metrics = evaluate(config, model, val_loader, epoch)
            writer.add_scalar("val/mIoU", metrics["mIoU"], epoch + 1)
            for cls, iou in metrics["class_iou"].items():
                writer.add_scalar(f"val/IoU_{cls}", iou, epoch + 1)

        # 모델 저장
        if (epoch + 1) % config.save_interval == 0:
            save_path = os.path.join(output_dir, f"weights/epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))

    progress_bar.close()


def evaluate(config, model, val_loader, epoch):
    """Evaluate MinkUNet on validation set."""
    model.eval()
    num_classes = config.model.backbone.out_channels
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating epoch {epoch+1}"):
            locs, features, labels, head_id = batch
            locs, features, labels = locs.cuda(), features.cuda(), labels.cuda()

            # Sparse Tensor 생성 및 예측
            sinput = ME.SparseTensor(features=features, coordinates=locs, device="cuda")
            output = model(sinput).F
            preds = output.argmax(dim=1)

            # Confusion matrix 업데이트
            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if label != config.ignore_index:
                    confusion_matrix[label, pred] += 1

    # mIoU 계산
    iou_per_class = []
    class_names = config.data.names
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        iou = intersection / union if union > 0 else 1.0
        iou_per_class.append(iou)

    mIoU = np.mean(iou_per_class)
    class_iou = {f"{class_names[i]}": iou for i, iou in enumerate(iou_per_class)}
    print(f"Epoch {epoch+1} - mIoU: {mIoU:.4f}, Class IoU: {class_iou}")

    return {"mIoU": mIoU, "class_iou": class_iou}


if __name__ == "__main__":
    # Config 로드
    config = OmegaConf.load("./config/semantic_seg_scannet.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.seed)
    train(config)