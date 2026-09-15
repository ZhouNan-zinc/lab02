# ConvLSTM 弹跳小球预测实验


## 环境

```bash
python3 -m venv venv_lab02
source venv_lab02/bin/activate
python -m pip install -r requirements.txt
```

脚本默认自动选择 CUDA；也可显式传入 `--device cpu` 或 `--device cuda`。

## 验证和运行

```bash
source venv_lab02/bin/activate
pytest -q
python convlstm_bounce.py --experiment baseline --smoke
python convlstm_bounce.py --experiment all
```

单独重跑某组：

```bash
python convlstm_bounce.py --experiment exp3
python convlstm_bounce.py --experiment best
```


## 数据切分说明

指南正文要求 2000 条训练序列和 200 条测试序列。参考代码仅生成 2000 条且用最后 200 条同时参与训练和测试，会造成数据泄漏。本实现按正文生成 2200 条，并使用 `[0,2000)` 训练、`[2000,2200)` 测试，二者严格互斥。

## 参数量核对

- 基线 ConvLSTM 细胞：38,144
- 输出卷积：289
- 基线总参数量：38,433
- 隐藏通道 64 的单细胞：150,016（约为 3.93 倍）
- FlattenLSTM：1,575,936（包含 PyTorch LSTM 的两组偏置）

预测图片为显示效果会把预测像素裁剪到 `[0,1]`；所有 MSE/MAE 均根据未经裁剪的模型原始输出计算。