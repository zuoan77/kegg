## 环境配置
```bash
# 创建虚拟环境
python -m venv kegg_env
source kegg_env/bin/activate  # Linux/Mac
# 或 kegg_env\Scripts\activate  # Windows

# 安装所有依赖
pip install -r requirements.txt
```

## 运行项目

1. 数据预处理
   ```bash
   bash DataPreProcess.sh
   ```

2. 抽象图构建
   ```bash
   bash AbstractGraphBuilder.sh
   ```

3. 模型训练
   ```bash
   bash StartTrain.sh
   ```

4. 模型预测
   ```bash
   bash StartPredict.sh
   ```