## F5-TTS Tensorrt-LLM Faster

为 F5-TTS 进行推理加速，测试样例如下：

+ `NVIDIA GeForce RTX 3090`
+ 测试文本为: `这点请您放心，估计是我的号码被标记了，请问您是沈沈吗？`

经测试，推理速度由`3.2s`降低为`0.72s`, 速度提升4倍！

基本的思路就是先将 `F5-TTS` 用 `ONNX` 导出，然后使用 `Tensorrt-LLM` 对有关 `Transformer` 部分进行加速。

特别感谢以下两个开源项目的贡献：
+ https://github.com/DakeQQ/F5-TTS-ONNX
+ https://github.com/Bigfishering/f5-tts-trtllm

全部`ref`见文档末尾。


> 项目构建耗时约 **3h**, 建议在 tmux 中构建。


## Install

```shell
conda create -n f5_tts_faster python=3.10 -y
source activate f5_tts_faster
```

```shell
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### F5-TTS 环境

```shell
# huggingface-cli download --resume-download SWivid/F5-TTS --local-dir ./F5-TTS/ckpts
git clone https://github.com/SWivid/F5-TTS.git   # 0.3.4
cd F5-TTS
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```shell
# 修改源码加载本地的 vocoder
vim /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/f5_tts/infer/infer_cli.py
# 如果是源码安装，则在 F5-TTS/src/f5_tts/infer/infer_cli.py
# 大约在124行，将 vocoder_local_path = "../checkpoints/vocos-mel-24khz" 注释，改为本地路径：
# vocoder_local_path = "/home/wangguisen/projects/tts/f5tts_faster/ckpts/vocos-mel-24khz"

# 运行 F5-TTS 推理
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "./assets/wgs-f5tts_mono.wav" \
--ref_text "那到时候再给你打电话，麻烦你注意接听。" \
--gen_text "这点请您放心，估计是我的号码被标记了，请问您是沈沈吗？" \
--vocoder_name "vocos" \
--load_vocoder_from_local \
--ckpt_file "./ckpts/F5TTS_Base/model_1200000.pt" \
--speed 1.2 \
--output_dir "./output/" \
--output_file "f5tts_wgs_out.wav"
```


### F5-TTS-Faster 环境：
```shell
conda install -c conda-forge ffmpeg cmake openmpi -y

# 为 OpenMPI 设置 C 编译器, 确保安装了 ggc
# conda install -c conda-forge compilers
# which gcc
# gcc --version
export OMPI_CC=$(which gcc)
export OMPI_CXX=$(which g++)

pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```python
# 检查 onnxruntime 是否支持 CUDA
import onnxruntime as ort
print(ort.get_available_providers())
```

### TensorRT-LLM 环境
```shell
sudo apt-get -y install libopenmpi-dev
pip install tensorrt_llm==0.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证 TensorRT-LLM 环境安装是否成功：
```python
python -c "import tensorrt_llm"
python -c "import tensorrt_llm.bindings"
```


## 转 ONNX

将 `F5-TTS` 导出到 `ONNX`:

```python
python ./export_onnx/Export_F5.py
```

```python
# 单独以 onnx 进行推理
python ./export_onnx/F5-TTS-ONNX-Inference.py
```

导出 `ONNX` 结构如下：
```shell
./export_onnx/onnx_ckpt/
├── F5_Decode.onnx
├── F5_Preprocess.onnx
└── F5_Transformer.onnx
```

转成 onnx 后，用 GPU 推理其实速度并没有很快，感兴趣的可以参考：
```python
# 指定 CUDAExecutionProvider
import onnxruntime as ort
sess_options = ort.SessionOptions()
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"], sess_options=sess_options)

# 量化：
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_quant.onnx", weight_type=QuantType.QInt8)
```

注意：因为导出 ONNX 的时候修改了 `F5` 和 `vocos` 的源码，所以需要重新 re-download 和 re-install，才能继续保证 F5-TTS 可用。
```shell
pip uninstall -y vocos && pip install vocos -i https://pypi.tuna.tsinghua.edu.cn/simple
```
```python
# ./export_trtllm/origin_f5 即为 F5 对应源码
cp ./export_trtllm/origin_f5/modules.py ../F5-TTS/src/f5_tts/model/
cp ./export_trtllm/origin_f5/dit.py ../F5-TTS/src/f5_tts/model/backbones/
cp ./export_trtllm/origin_f5/utils_infer.py ../F5-TTS/src/f5_tts/infer/
```


## 转 Trtllm

### 源码改动

装好 `TensorRT-LLM` 后，所以需要移动目录：

本项目目录中 `export_trtllm/model` 对应 `Tensorrt-LLM` 源码中的 `tensorrt_llm/models`。

1. 在 `Tensorrt-LLM` 源码中的 `tensorrt_llm/models` 目录下新建 `f5tts` 目录，然后将 repo 中的代码放入对应的目录。

```shell
# 查看 tensorrt_llm
ll /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages | grep tensorrt_llm

# 源码 tensorrt_llm/models 导入
mkdir /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts

cp -r /home/wangguisen/projects/tts/f5tts_faster/export_trtllm/model/* /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts
```

+ `tensorrt_llm/models/f5tts` 目录如下:
```shell
/home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts/
├── model.py
└── modules.py
```

2. 在 `tensorrt_llm/models/__init__.py `导 `入f5tts`:

```shell
vim /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/__init__.py
```

```python
from .f5tts.model import F5TTS

__all__ = [..., 'F5TTS']

# 并且在 `MODEL_MAP` 添加模型：
MODEL_MAP = {..., 'F5TTS': F5TTS}
```


### convert_checkpoint
```python
python ./export_trtllm/convert_checkpoint.py \
        --timm_ckpt "./ckpts/F5TTS_Base/model_1200000.pt" \
        --output_dir "./ckpts/trtllm_ckpt"

# --dtype float32
```

### build engine
> 支持Tensor 并行, --tp_size

```python
trtllm-build --checkpoint_dir ./ckpts/trtllm_ckpt \
             --remove_input_padding disable \
             --bert_attention_plugin disable \
             --output_dir ./ckpts/engine_outputs
# 如报参数dtype不一致错误，那是因为我们默认参数是fp16，而网络参数默认需要fp32，需要在tensorrt_llm/parameter.py中将参数默认_DEFAULT_DTYPE = trt.DataType.HALF
```

## fast inference
```python
python ./export_trtllm/sample.py \
        --tllm_model_dir "./ckpts/engine_outputs"
```



## References

https://github.com/SWivid/F5-TTS

https://github.com/DakeQQ/F5-TTS-ONNX

https://github.com/NVIDIA/TensorRT-LLM

https://github.com/Bigfishering/f5-tts-trtllm