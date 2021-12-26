# Nested UNet (UNet++) for Building Detection

## Demo

| <img src="img/val_img.png" alt="val_img" width="400px"> | <img src="img/val_label.png" alt="val_label" width="400px"> | <img src="img/output.png" alt="output" width="400px"> | <img src="img/CRF.png" alt="CRF" width="400px"> | <img src="img/denoised.png" alt="denoised" width="400px"> |
|:-------------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| Input Image | Ground Truth | UNet++ Output | Post-prosessed by Fully Connected CRF |  Post-prosessed by Fully Connected CRF and Denoising |

## Usage

1. Install

    ```bash
    $ git clone https://github.com/hashi0203/Nested-UNet-Building-Detection.git
    $ cd Nested-UNet-Building-Detection
    $ pip install -r requirements.txt
    ```

2. Prepare data

    <pre>
    data
    ├── test_img
    │   ├── xxx.png
    │    &#65049;
    │   └── xxx.png
    ├── train_img
    │   ├── xxx.png
    │    &#65049;
    │   └── xxx.png
    ├── train_label
    │   ├── xxx.png
    │    &#65049;
    │   └── xxx.png
    ├── val_img
    │   ├── xxx.png
    │    &#65049;
    │   └── xxx.png
    └── val_label
        ├── xxx.png
         &#65049;
        └── xxx.png
    </pre>

3. Edit `config.py`

4. Train

    ```bash
    $ mkdir graph
    $ python train.py
    ```

5. Edit `config.py`

    Set the model to use for evaluation

6. Evaluate

    Warning: Denoising may take a long time

    ```bash
    $ mkdir result
    $ python evaluate.py    # without post processing
    $ python evaluate.py -c # with post processing (Fully Connected CRF)
    $ python evaluate.py -d # with post processing (Fully Connected CRF and denoising)
    ```
