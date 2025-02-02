# リポジトリ概要
Lightningを用いてLoRAを訓練するスクリプト群

jsonファイルにハイパーパラメータを書き込み、その条件に基づいた訓練を行う

## 環境構築
環境構築には**Docker**を用いる。

具体的には、このリポジトリに用意されている`./shell`ディレクトリ内のシェルスクリプトを用いることでDockerfileからImageの作成、Containerの立ち上げが行える。

### 実行準備
`./shell`内のシェルスクリプトに実行権限の付与を行う。
```
chmod +x ./shell/*.sh
```

### DockerImageの作成
`./shell`内の`build_image.sh`を用いてImageの作成を行う。
```
./shell/build_image.sh {image_name}
```
引数に任意のイメージ名を指定することができるが、指定しなくても問題なく動作する。指定がない場合は、デフォルトで**private_lora_image**という名前でImageが作成される。

### DockerContainerの立ち上げ
`./shell`内の`run_container.sh`を用いてContainerの立ち上げ及び、環境内への接続を行う。
```
./shell/run_container.sh {image_name} {container_name}
```
引数に任意のイメージ名やコンテナ名を指定することができるが、指定しなくても問題なく動作する(※Imageの名称はbuild_image.shで作成したイメージ名と一致させる)。

指定がない場合は、デフォルトでそれぞれ**private_lora_image**、**private_lora_container**という名前で実行される。

また、ここでコンテナに割り当てるGPUやRAMを調整したい場合は直接シェルスクリプトの書き換えを行う
```
# 修正例
docker container run -dit --rm \
    --name ${container_name} \
    --memory=32g \ # RAMの増加
    --shm-size=16g \ # 共有RAMの増加
    --gpus '"device=0,1"' \ # gpuデバイスの増加
    ${image_name}
```


### コンテナ内部でssh鍵とGithubアカウントの連携
こちらの[記事](https://qiita.com/shizuma/items/2b2f873a0034839e47ce)を参照しながら、**コンテナ内部で**ssh鍵の作成、Githubアカウントへの登録、Configの設定を行う。

### Container内で実行環境の構築
コンテナ内部で再度、githubからこのリポジトリのCloneを行う
```
git clone git@github.com:h-hosoda-ml/Private-LoRA-Lightning.git
```

Cloneしたリポジトリの中へ移動し、pipを用いてライブラリのダウンロードを行う。
```
cd Private-LoRA-Lightning
[任意] python -m venv .venv (仮想環境の軌道)
[任意] source .venv/bin/activate (仮想環境の有効化)
pip install -r requirements_pip.txt
```
上記のようにして実行環境がコンテナ内部で立ち上がる

## Training LoRA
訓練の設定として指定可能な引数は`lora_lightning/arguments.py`内の `TrainingArgs` クラスに格納されている。代表的な引数を以下に示す。

`model` : str => モデルの名称 (Hugging Faceでの使用を想定)

`num_train_epochs` : int => 訓練Epoch数

`precision` : Literal [str] => モデルを表現する数値精度

`lora_rank`: int => loraのランク数

`modify_layers`: str => loraの変更を加えるLayer名

これらの変更はコマンドライン上で反映するためには、jsonファイルの書き換えを行い **-c オプション**にjsonファイルを指定する。

### <実行例>
```
python train_lora.py -c conf/sample.json
```
