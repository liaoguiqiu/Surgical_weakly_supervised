debug_params = {
    # 'model': 'deit_small_MCTformerPlus',
    'model':  'deit_small_MCTformerV2_patch16_224',
    'batch_size': 5,
    # 'data_set': 'VOC12MS',
    'data_set': 'VOC12',

    'img_list': 'voc12',
    'data_path': 'C:/2data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',
    'layer_index': 12,
    'output_dir': '/MCTformer_results/MCTformer_v1',
    'finetune': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
    # 'gen_attention_maps': True,
    # 'resume': '/MCTformer_results/MCTformer_v1/checkpoint.pth',
    # 'img_list': 'C:/2data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation',
    # Add other parameters here...
}
