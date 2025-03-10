# DA6401 Assignment 1

## Setup Instructions for the Examiner

If this is being run on a **new system**, please follow these steps to ensure the environment is correctly prepared:

---

### 1️⃣ Install Python 3.11
Download and install Python 3.11 from the official website:  
https://www.python.org/downloads/release/python-3110/

---

### 2️⃣ Install required Python packages
After installing Python 3.11, open **Command Prompt (CMD)** and run:

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Login to Weights & Biases (WandB)
If WandB is not already logged in on the device, execute:

```bash
wandb login
```
You can get the API key from https://wandb.ai/authorize.
my api key --- ``` 3d199b9bde866b3494cda2f8bb7c7a633c9fdade ```

---

### 4️⃣ Running the Training Script
To execute the training, use the following **command**:
### change epoch according like current set to 10
```bash
python train.py -wp DA6401_Assignment_1 -we cs22m088 -d fashion_mnist -e 10 -b 64 -l cross_entropy -o nadam -lr 0.001 -m 0.9 -beta 0.95 -beta1 0.9 -beta2 0.999 -eps 1e-8 -w_d 0 -w_i he_uniform -sz 256 -nhl 5 -a relu
```

This will:
- Train the model on the **Fashion MNIST dataset**.
- Run the training for **10 epochs**.
- Track results via **Weights & Biases** if logged in.

---
### Link to WandB Report
-``` https://wandb.ai/cs22m088-iit-madras/DA6401_Assignment_1/reports/DA6401-Assignment-1--VmlldzoxMTcwNTQ5NQ?accessToken=cxp3welccil72eizsj1uobhoywj7udesu77zz8m7cxlrshynmkogm9zm2lycp3zq  ```

### Link to Github Repo-
- ```  https://github.com/Swapnil7-lab/DA6401_Assignment_1 ```


### Optional:
If the **Fashion MNIST sample image** pops up and pauses execution, simply **close the window** to allow training to continue.

---

### Notes:
- Python 3.12 is **NOT recommended**. Use **Python 3.11** for compatibility.
- Ensure internet access is available for downloading datasets on first run.
- Make sure your system has sufficient memory and CPU for smooth execution.

---

Prepared by: *** Swapnil-CS22M088 ***
