## Description

This project defines a data pipeline for [HistoGPT](https://github.com/marrlab/HistoGPT/) model that generates diagnostic text descriptions from large digital pathology image.

The configuration should specify:
  - the folder of the images
  - the file ending to consider (e.g. '.svs.' or '.ndpi')
  - whether to save the image patches for debugging in the output folder

The pipeline consists of 3 steps:
  - Runs the embedding model to generate embeddings for the patches of the images. Here all images from the configured folder which match the configured file ending is processed.
  - Uses the generated embeddings to run the transformer model to generate the clinical report for each image and store the clinical report as a .txt file in the output folder.
  - Aggregates all the .txt files into a .csv file which is stored as result.csv in the output folder.

## How to use

Navigate to the project folder after cloning and build the Docker image:

```
docker-compose build
```

Start the container:

```
docker-compose up
```

This starts the application and launches the Dagster UI that can be accessed on port [3000](http://127.0.0.1:3000/).
