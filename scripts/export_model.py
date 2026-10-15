from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

output_file = Path('models/mobilenet_v2.onnx')


def main():
    if output_file.exists():
        print(f'{output_file} already exists')
        return
    output_file.parent.mkdir(exist_ok=True)

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model=model,
        args=(dummy_input,),
        f=str(output_file),
        input_names=['input'],
        output_names=['predictions'],
    )
    print(f'Model exported to {output_file}')


if __name__ == '__main__':
    main()
