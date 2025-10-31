#!/bin/bash

echo "开始抽象图构建..."
cd src/graph_construction

echo "生成抽象节点连接..."
python generate_abstract_connections.py



echo "构建加权图并生成可视化..."
python main.py

echo "抽象图构建完成！"



