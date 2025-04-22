## F5-TTS Tensorrt-LLM Faster

Inference acceleration for F5-TTS. A sample test is as follows:

+ `NVIDIA GeForce RTX 3090`
+ Test text: `Please don’t worry about this. My number was probably flagged. Are you Shen Shen?`

After testing, the inference time was reduced from`3.2s`to`0.72s`, achieving a 4x speed increase!

The basic approach is to first export `F5-TTS` using `ONNX` and then use `Tensorrt-LLM` to accelerate the `Transformer` components.

Special thanks to the following open-source projects for their contributions:：
+ https://github.com/DakeQQ/F5-TTS-ONNX
+ https://github.com/Bigfishering/f5-tts-trtllm

All references are listed at the end of the documentation.

**Model Weights**：https://huggingface.co/wgs/F5-TTS-Faster

> Project build time is approximately **3h**,  so it is recommended to build it in tmux


## Install

```shell
conda create -n f5_tts_faster python=3.10 -y
source activate f5_tts_faster
```

```shell
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### F5-TTS Environment

```shell
# huggingface-cli download --resume-download SWivid/F5-TTS --local-dir ./F5-TTS/ckpts
git clone https://github.com/SWivid/F5-TTS.git   # 0.3.4
cd F5-TTS
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```shell
# Modify the source code to load the local vocoder
vim /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/f5_tts/infer/infer_cli.py
# If installed from source, the file will be at: F5-TTS/src/f5_tts/infer/infer_cli.py
# Around line 124, comment out vocoder_local_path = "../checkpoints/vocos-mel-24khz" 
# and change it to your local path, for example:
# vocoder_local_path = "/home/wangguisen/projects/tts/f5tts_faster/ckpts/vocos-mel-24khz"

# Run F5-TTS inference
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "./assets/wgs-f5tts_mono.wav" \
--ref_text "I'll call you back then, please make sure to answer." \
--gen_text "Please don’t worry about this. My number was probably flagged. Are you Shen Shen?" \
--vocoder_name "vocos" \
--load_vocoder_from_local \
--ckpt_file "./ckpts/F5TTS_Base/model_1200000.pt" \
--speed 1.2 \
--output_dir "./output/" \
--output_file "f5tts_wgs_out.wav"
```


### F5-TTS-Faster Environment：
```shell
conda install -c conda-forge ffmpeg cmake openmpi -y

# Set the C compiler for OpenMPI, make sure gcc is installed
# You can install compilers like this:
# conda install -c conda-forge compilers
# Check the path and version of gcc
# which gcc
# gcc --version
export OMPI_CC=$(which gcc)
export OMPI_CXX=$(which g++)


# Install Python dependencies
pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```python
# Check if onnxruntime supports CUDA
import onnxruntime as ort
print(ort.get_available_providers())
```

### TensorRT-LLM Environment
```shell
sudo apt-get -y install libopenmpi-dev
pip install tensorrt_llm==0.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Verify that TensorRT-LLM is installed correctly:
```python
python -c "import tensorrt_llm"
python -c "import tensorrt_llm.bindings"
```


##Exporting to ONNX

To export `F5-TTS` to the `ONNX`format:

```python
python ./export_onnx/Export_F5.py
```

```python
# Run inference using ONNX directly
python ./export_onnx/F5-TTS-ONNX-Inference.py
```

The exported ONNX model structure looks like this:
```shell
./export_onnx/onnx_ckpt/
├── F5_Decode.onnx
├── F5_Preprocess.onnx
└── F5_Transformer.onnx
```

After converting to ONNX, GPU inference isn’t super fast by default. If you're interested, try this optimization:
```python
# Specify CUDAExecutionProvider for ONNX Runtime
import onnxruntime as ort
sess_options = ort.SessionOptions()
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"], sess_options=sess_options)

# Quantize the ONNX model
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_quant.onnx", weight_type=QuantType.QInt8)
```

 Note: Since exporting to ONNX requires modifying the source code of both `F5` and `vocos` , you’ll need to re-download and re-install them to keep F5-TTS functional:
```shell
pip uninstall -y vocos && pip install vocos -i https://pypi.tuna.tsinghua.edu.cn/simple
```
```python
# ./export_trtllm/origin_f5 contains the modified F5 source files
cp ./export_trtllm/origin_f5/modules.py ../F5-TTS/src/f5_tts/model/
cp ./export_trtllm/origin_f5/dit.py ../F5-TTS/src/f5_tts/model/backbones/
cp ./export_trtllm/origin_f5/utils_infer.py ../F5-TTS/src/f5_tts/infer/
```


## Exporting to TensorRT-LLM

### Source Code Changes

After installing `TensorRT-LLM`, you’ll need to move some directories to integrate the model.

the`export_trtllm/model` folder from this repo should be moved to the `tensorrt_llm/models` directory inside the TensorRT-LLM Python package.

1. First, create a new directory for `f5tts` inside `tensorrt_llm/models`, then copy the code there:

```shell
# Find the path to the installed tensorrt_llm
ll /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages | grep tensorrt_llm

# Create a directory inside the TensorRT-LLM models path
mkdir /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts

# Copy the exported model files into it
cp -r /home/wangguisen/projects/tts/f5tts_faster/export_trtllm/model/* /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts
```

+ The `tensorrt_llm/models/f5tts` directory should look like this:
```shell
/home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts/
├── model.py
└── modules.py
```

2. Import `f5tts` in `tensorrt_llm/models/__init__.py`:

```shell
vim /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/__init__.py
```

```python
from .f5tts.model import F5TTS

__all__ = [..., 'F5TTS']

# Also add the model to `MODEL_MAP`:
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
> Supports tensor parallelism with --tp_size

```python
trtllm-build --checkpoint_dir ./ckpts/trtllm_ckpt \
             --remove_input_padding disable \
             --bert_attention_plugin disable \
             --output_dir ./ckpts/engine_outputs
# If you get a data type mismatch error, it’s likely because the default parameter is fp16, but the network expects fp32. To fix this, edit tensorrt_llm/parameter.py and set the default:
```
```python
_DEFAULT_DTYPE = trt.DataType.HALF
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
