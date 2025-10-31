#!/bin/bash

echo "开始数据预处理..."
cd src/data_processing
python kegg_processor.py
echo "数据预处理完成！"



