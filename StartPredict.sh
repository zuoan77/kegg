#!/bin/bash

# 默认的KO序列
DEFAULT_KO_SEQUENCE="K05825 K00172 K00189 K00171 K18358 K00169 K00170 K00929 K00634"

# 如果环境变量中已设置KO_SEQUENCE，则使用环境变量的值
if [ -z "$KO_SEQUENCE" ]; then
    # 提示用户输入
    echo "请输入KO序列（多个KO值用空格分隔）"
    echo "直接回车将使用默认值："
    echo "$DEFAULT_KO_SEQUENCE"
    echo -n "> "
    read input_sequence
    
    # 如果用户直接回车，使用默认值
    if [ -z "$input_sequence" ]; then
        KO_SEQUENCE="$DEFAULT_KO_SEQUENCE"
        echo "使用默认KO序列"
    else
        KO_SEQUENCE="$input_sequence"
        echo "使用输入的KO序列"
    fi
fi

# 运行推理脚本
echo "开始进行路径推理..."
cd src/kegnn_model
python ko_node_predictor.py --ko_sequence "$KO_SEQUENCE"
