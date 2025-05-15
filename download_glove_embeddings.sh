#!/bin/sh

cd data && \
curl -L -o embeddings.zip https://nlp.stanford.edu/data/glove.6B.zip && \
unzip embeddings.zip && \
rm embeddings.zip
