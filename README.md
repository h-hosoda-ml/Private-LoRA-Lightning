# リポジトリ概要
Lightningを用いてLoRAを訓練するスクリプト群

jsonファイルにハイパーパラメータを書き込み、その条件に基づいた訓練を行う

## 環境構築
環境構築には `rye` を用いることが前提となっている

インストールは以下のドキュメントを参考にすること。

[rye installation](https://rye.astral.sh/guide/installation/)

インストールが完了後、private-lora-routing/ 配下で以下のコマンドを実行

```
rye sync
source .venv/bin/activate
```

以上の処理によって、外部ライブラリのインストール、仮想環境の有効化が行える

## Training LoRA
訓練の設定として指定可能な引数は `TrainingArgs` クラスに格納されている

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

