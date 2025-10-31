#!/bin/bash

echo "开始训练SimpleKEGNN模型..."
cd src/kegnn_model

echo "数据处理"
python process_kegg_data.py
echo "数据处理完成"


echo "训练SimpleKEGNN模型..."
python train_kegg_simple.py

echo "SimpleKEGNN模型训练完成！"




