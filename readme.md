# Anomaly Detection Pipeline



<img src="docs/example_anomaly_maps.png" width="500"/>

This repo implements an anomaly detection pipeline using Patchcore and other anomaly detection models :


---

## Install

```shell
# Create a new conda environment using python 3.6.13 using below command

$ conda create -n test_env python==3.6.13

# Install other dependencies using requirements.txt file

$ pip install -r requirements.txt

# Next install torch , i used torch 1.9.0 + cu111

$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Your environment is set now!
```

## Usage

CLI:
```shell
To run the code just use below command

$ python generate_results.py
```
This command will generate 3 folders : 

    1 - {dataset}_heatmap : The predictions heatmap from patchcore model
    2 - {dataset}_res : The corresponding masks for the anomaly part
    3 - cropped_{dataset} : The cropped anomaly part

-> The code is setup using road dataset that can be found under /datasets/road directory.

-> To setup your dataset and train patchcore put you dataset (say "hello" is the name of your dataset) in the /datasets/ directory

-> Now go to indad/new_data/ folder and add your dataset folder name to the function mvtec_classes()

Follow the below folder structure : Place normal images under train/good/  and abnormal images under test/defect/

---road-- <br>
         | <br>
         |__train---  <br>
         |          |___good-------img1.   <br>
         |                      |___img2..   <br>
         |___test.  <br>
                |____defect---img1.   <br>
                           |___img2...   <br>
         


---



## References

SPADE:
```bibtex
@misc{cohen2021subimage,
      title={Sub-Image Anomaly Detection with Deep Pyramid Correspondences}, 
      author={Niv Cohen and Yedid Hoshen},
      year={2021},
      eprint={2005.02357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

PaDiM:
```bibtex
@misc{defard2020padim,
      title={PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization}, 
      author={Thomas Defard and Aleksandr Setkov and Angelique Loesch and Romaric Audigier},
      year={2020},
      eprint={2011.08785},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

PatchCore:
```bibtex
@misc{roth2021total,
      title={Towards Total Recall in Industrial Anomaly Detection}, 
      author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
      year={2021},
      eprint={2106.08265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
