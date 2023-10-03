# CLIPテスト運用
画像 ↔ テキスト の変換モデル**CLIP**のテストを行いました。CLIPはラベルの自由度が高い自然言語教師型画像分類モデルです。

環境構築、テストコードを記載します。

CLIPのgithubのReadMe（[こちら](https://github.com/openai/CLIP "CLIP github")）を参考にして進めました。

### 環境構築
Pytorch1.7.1以上と、torchvisionが必要です。（Pythonはすでに入ってるものとします。ver3.9.6で動作を確認しています）

ここではWindowsのVSコード上でPowerShellを使って進めていきます。

PyTorch、torchvisionをインストールします。
```powershell
$ pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

tqdmとCLIPをインストールします。

```powershell
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

その他バージョン・OSごとのインストールは[こちら](https://pytorch.org/get-started/locally/ "PyTorch GET STARTED")。


### テストコード
入力画像を用意します。
![bedroom](/image/e10d6b636ed94caa87771f92b1251868_i2_1.jpg)

Pythonで動作させます（CIFAR100というデータセットをダウンロードして使用するので、最初は時間かかります）
```python
import os
import clip
import torch
from PIL import Image
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image = Image.open("../clip/image/e10d6b636ed94caa87771f92b1251868_i2_1.jpg")
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

結果
```
Top predictions:

             bed: 84.44%
           couch: 13.17%
           chair: 0.69%
           table: 0.50%
            lamp: 0.18%
```