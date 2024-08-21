# LLaMA-Factory Document

Document for https://github.com/hiyouga/LLaMA-Factory

Visit https://llamafactory.readthedocs.io/ for the document.

## Contribution

Doc contribution welcome. Before creating a PR, please check and test your docs locally as follows:

1. Step into the path `docs`:

```shell
cd docs
```

2. Install the required dependencies:

```shell
pip install -r requirements.txt
```

3. Build

```shell
make html
```

4. Server on localhost for doc preview:

```shell
python -m http.server -d build/html 8008
```

## Acknowledgement

This repo benefits from [Qwen2](https://github.com/QwenLM/Qwen2). Thanks for their wonderful works.
