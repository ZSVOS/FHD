# dataset settings
dataset_type = 'LEVIRPlusDataset'
data_root_LEVIR = './data/LEVIR+/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, map_255_to_1=True),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortionMultiImages'),
    dict(type='NormalizeMultiImages', **img_norm_cfg),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='NormalizeMultiImages', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

LEVIR_train = dict(
    type=dataset_type,
    data_root=data_root_LEVIR,
    img1_dir='train/A',
    img2_dir='train/B',
    ann_dir='train/label',
    pipeline=train_pipeline)
    
LEVIR_val = dict(
    type=dataset_type,
    data_root=data_root_LEVIR,
    img1_dir='val/A',
    img2_dir='val/B',
    ann_dir='val/label',
    pipeline=test_pipeline)

LEVIR_test = dict(
    type=dataset_type,
    data_root=data_root_LEVIR,
    img1_dir='test/A',
    img2_dir='test/B',
    ann_dir='test/label',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=[LEVIR_train],
    val=LEVIR_val,
    test=LEVIR_test
    )

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='./model_ckpt/mit_b2.pth',
    backbone=dict(
        type='FHD',
        ori_type='mit_b2',
        style='pytorch'),
    decode_head=dict(
        type='FHD_Head',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
# optimizer
optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=999999, metric=['mFscore', 'mFscoreCD'])
optimizer_config = dict()


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook', interval=10)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

