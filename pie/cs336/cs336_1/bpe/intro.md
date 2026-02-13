Example Usage:
```bash
python ./bpe.py --num_processes 16 --special_tokens ./special_tokens.txt --train_dataset ../data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 --output_merges ./TinyStories_merges.pkl --output_vocab ./TinyStories_vocab.pkl
```
Example Usage:
```bash
python ./bpe.py --num_processes 16 --special_tokens ./special_tokens.txt --train_dataset ../data/owt_train.txt --vocab_size 32000 --output_merges ./owt_merges.pkl --output_vocab ./owt_vocab.pkl
```
