# RUIP
We provide a sample code of the RUIP method for defending against StarGAN.

## Usage
### Installation
1. Prepare the Environment
   Install the lib by pip (recommend)
    ```
    pip install -r requirements.txt
    ```
2. Prepare the Datasets
   Download the [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) datasets
3. Prepare the Model Weights
   
   For your convenient usage, we prepare the weights download link in [Baiduyun disk](https://pan.baidu.com/s/1YPPlCbeR0teZPuMbprGMPQ)
   
   Extract code: yxjr.

   The name of stargan generator model weights is *200000-G.ckpt*. We also prepare a RUIP model for you to test its performance named *ruip.tar*.

   ### Inference
    ```
    python test.py
    ```
