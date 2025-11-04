```bash

conda create -n myenv python=3.7 -y

conda activate myenv

conda install pytorch==1.9.0 torchtext==0.10.0
 
conda install spacy=3.0.0 -c conda-forge

python -m spacy download en_core_web_sm

python -m spacy download de_core_news_sm
```
