# BounWiki: A transformer-based QA-Engine for Bogazici University

This is the official repository of my report [**BounWiki: A transformer-based QA-Engine for Bogazici University**](./JointXplore.pdf) by Leonard Schenk for the course Transformer Based Pretraining in Natural Language Processing.
## Abstract
In this research, we propose the use of a transformer-based neural network for answering questions about Bogazici University. The proposed architecture utilizes a context embedding space to retrieve relevant documents based on similarity to an initial query and employs reader models to extract or generate the corresponding answer. We also conduct a comprehensive study comparing various models and perform experiments with different subsets of the context and validation dataset to evaluate the strengths and weaknesses of our approach. Results indicate that larger, more capable models tend to perform better, but still struggle with certain aspects of the context and validation dataset. To facilitate further research on this topic, besides this codebase, a demo version can be accessed on [Huggingface](https://huggingface.co/spaces/LeoGitGuy/BounWiki).
## Rerunning Evaluation Experiments

A tutorial on how to run the experiments is explained in the jupyter notebook [BounWikiExperiments](./BounWikiExperiments.ipynb).
Alternatively, they experiments can also be executed in a standard python environment by following these steps:

### Installation
Install requirements with:
```shell
pip install -r requirements.txt
```
### Usage

The code can be run with the following command:
```shell
python run.py --context "processed_website_text" "processed_website_tables" 
               --text_reader "bert-base" 
               --table_reader "tapas" 
               --api-key ""
               --seperate_evaluation
               --top_k 1
```

for more options and detailed information on each option please run `python run.py -h`

## Running the demo app

#### Option 1: Gradio

Use Huggingface to directly host gradio applications (see [here](https://huggingface.co/docs/hub/spaces-sdks-gradio)). To do this create a new gradio space, add all files from this repository and the `app.py` file will automatically load the application. Alternatively this file can also be used to build a docker container or the app from huggingface can be embedded into other websites as iframe (see [here](https://huggingface.co/docs/hub/spaces-sdks-gradio#embed-gradio-spaces-on-other-webpages))

#### Option 2: Docker
Another repository was created at [huggingface/BounWiki](https://huggingface.co/spaces/LeoGitGuy/BounWiki/tree/main), which contains a Dockerfile, a main.py file using Fastapi and a static folder with HTML, CSS and Javascript for serving the application. If you want to modify this, you can clone it directly from there.
It is already hosted on a [docker space](https://huggingface.co/spaces/LeoGitGuy/BounWiki) in huggingface.
