# Task000
## `vllm` related 
vLLM is straightforward to install. The recommended method is to use pip with the following command:
```bash
pip install vllm
```
Python version 3.10 is recommended due to its compatibility with many essential libraries.
To start the vLLM server, use the following command will launch a server in one node with 8 GPUs being utilized:
```bash
vllm serve PATH/TO/YOUR/MODEL --tensor-parallel-size 8
```
While this example demonstrates the --tensor-parallel-size parameter, vLLM offers numerous additional configuration options not covered here. By default, the server will utilize all available GPUs, with the specific count determined by the `--tensor-parallel-size` argument. At the time of writing, vLLM does not support data parallelism. Upon successful launch, GPU memory usage will immediately increase significantly, while GPU utilization may initially remain at 0% until inference requests are processed. If running in the foreground, the vLLM server can be terminated by pressing `Ctrl+C`.

## Model installation
Downloading models directly from Hugging Face can often be unacceptably slow. To mitigate this, we recommend using `modelscope` for model acquisition.
```bash
pip install modelscope
```
The installation process is straightforward and can be completed using:
```python
from modelscope import snapshot_download
snapshot_download(model_id = 'FIND/IT/FROM/MODELSCOPE/WEBPAGE', revision = 'master', cache_dir = 'PATH/TO/SAVING')
```
The downloaded model will be saved to the specified `cache_dir`. Navigate to this directory until you find the root containing essential files such as `config.json` and the tensor weight files. This root directory is the path that must be provided to the `vllm serve` command to locate and load your model correctly.

## Using http to obtain inference result
Different models may produce varying output formats. For instance, some models can generate `<think>` and `</think>` tags in their responses. This behavior can typically be configured through the HTTP request payload. The simplest method to construct a proper payload is to follow examples from official documentation. Here is a basic example using curl:
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 8192,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```
This example provides most of the essential configuration parameters. Note that: `http://localhost:8000/v1/chat/completions` is the default local endpoint for vLLM, which can be configured through command-line parameters when starting the server. `Content-Type: application/json` is the required HTTP header. The messages parameter is important, as chat-based models often support multiple roles:

```bash
"messages": [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content}
]
```
Below is a Python implementation demonstrating how to make HTTP requests with these settings:
```python
payload = {
    "model": model_name, # The name or path of a HuggingFace Transformers model.
    "messages": [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
        ],
        "temperature": t, # temperature controls distributions. Lower temperature means more sharped distribution.
        "top_p": top_p, # top_p selects tokens from the pools whose probability ranks in decreasing order until the total probability exceeds top_p
        "top_k": top_k, # top_k selects top_k tokens from the pools whose probability ranks in decreasing order
        "max_tokens": max_tokens,
        "stream": False # I not sure what this used
        "chat_template_kwargs": {"enable_thinking": true} # custom keywords and it does not follow the standard openAI API
    }
    headers = {"Content-Type": "application/json"}
    async with session.post(
        url=url,
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=time_out)
    ) as response:
        if response.status == 200:
            data = await response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return await response.text()
```
This approach supports concurrent HTTP requests for improved performance.
## Inference efficiency

