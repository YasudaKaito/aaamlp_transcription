FROM continuumio/miniconda3

WORKDIR /opt/study/aaamlp_transcription
ENV TZ=Asia/Tokyo
COPY environment.yml .

# ref. https://qiita.com/kimisyo/items/66db9c9db94751b8572b
# ref. https://pythonspeed.com/articles/activate-conda-dockerfile/
# activate できないため SHELL 命令を使っている
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "ml", "/bin/bash", "-c"]
