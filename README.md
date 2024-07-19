
# platedet

`platedet` is a lightweight package for detecting license plates in images using an ONNX model. It is designed to be part of a pipeline for detecting, cropping, and reading license plates. The underlying model is just YOLO, converted to ONNX format. The training data comprises primarily Brazilian license plates, sourced from internet images.

## Installation

To install the required dependencies, use the following command:

For cpu

```bash
pip install "platedet[cpu]"
```

For cuda 11.X
```bash
pip install "platedet[gpu]"
```

For cuda 12.X
```bash
pip install "platedet[gpu]" --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## Usage

### Command Line Interface

You can use the command line interface to detect license plates in an image:

```bash
platedet image_path [--model_path MODEL_PATH] [--return_types RETURN_TYPES] [--providers PROVIDERS] [--save_images] [--show_result]
```

#### Arguments

- `image_path`: Path to the input image.
- `--model_path`: Path to the ONNX model (default: `artifacts/platedet.onnx`).
- `--return_types`: Output formats (choices: `raw`, `pil`, `np_array`, `boxes`).
- `--providers`: ONNX Runtime providers (default: `CPUExecutionProvider`).
- `--save_images`: Save cropped images of detected plates.
- `--show_result`: Display results with bounding boxes and confidence scores.

### Example

To detect license plates and save the cropped images:

```bash
platedet path/to/image.png --save_images
```

To display the results with bounding boxes:

```bash
platedet detect.py path/to/image.png --show_result
```

### Using in Code

```python
from PIL import Image
from platedet import Platedet

platedet = Platedet()
image = Image.open('examples/1.jpg')
crops = platedet.inference(image, return_types=['pil'])
for idx, crop in enumerate(crops['pil']['images']):
    crop.save(f'{idx}.jpg')
```

If you want to use CUDA:
```python
from PIL import Image
from platedet import Platedet

platedet = Platedet(providers=["CUDAExecutionProvider"])
image = Image.open('examples/1.jpg')
crops = platedet.inference(image, return_types=['pil'])
for idx, crop in enumerate(crops['pil']['images']):
    crop.save(f'{idx}.jpg')
```

Check all execution providers [here](https://onnxruntime.ai/docs/execution-providers/).
