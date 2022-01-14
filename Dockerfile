FROM continuumio/miniconda3

WORKDIR /opt/study/aaamlp_transcription
COPY requirements.txt .

# ref. https://qiita.com/kimisyo/items/66db9c9db94751b8572b
# ref. https://pythonspeed.com/articles/activate-conda-dockerfile/
# activate できないため SHELL 命令を使っている
RUN conda create -n ml python==3.7.6
SHELL ["conda", "run", "-n", "ml", "/bin/bash", "-c"]
RUN pip install -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/opt/study/aaamlp_transcription/project/src"
ENTRYPOINT jupyter notebook \
    --notebook-dir=/opt/study/aaamlp_transcription/project --ip='*' --port=8888 \
    --no-browser --allow-root
