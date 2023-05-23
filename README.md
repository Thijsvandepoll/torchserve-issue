# Example to reproduce issue

First, install requirements and create a `model_store` directory:
```
pip install -r requirements.txt
mdkir model_store
```

Now, build and save the model + tokenizer:
```
python build.py
```

Archive the model:
```
torch-model-archiver --model-name custom_model --version 1.0 --serialized-file ./pytorch_model.bin --handler ./handler.py --extra-files "./config.json,./special_tokens_map.json,./tokenizer.json,./tokenizer_config.json,./vocab.txt" --requirements-file ./requirements.txt && mv custom_model.mar model_store
```

Serve the model in Docker:
```
docker run --rm -it -p 8080:8080 -p 8082:8082 -v $(pwd)/model_store:/home/model-server/model-store  pytorch/torchserve:latest-cpu torchserve --start --model-store model-store --models custom_model=custom_model.mar --ncs
```

Call the model to verify that it works. This should take a few seconds (slow). Please try it a couple of times:
```
curl -v -H "Content-Type: application/json" http://localhost:8080/predictions/custom_model --data '{"instances": ["This is an example sentence"]}'
```

Now we build the custom container:
```
docker build -t torchserve-custom:latest .
```

Serve the container:
```
docker run --rm -it -p 8090:8080 -p 8092:8082 -v $(pwd)/model_store:/model-store torchserve-custom:latest torchserve --start --model-store model-store --models custom_model=custom_model.mar --ncs --foreground
```

Now create a request. This should respond much quicker than the other. Please try it a couple of times:
```
curl -v -H "Content-Type: application/json" http://localhost:8090/predictions/custom_model --data '{"instances": ["This is an example sentence"]}'
```