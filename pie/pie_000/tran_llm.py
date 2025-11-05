import argparse
import aiohttp
import asyncio
import pandas as pd
import re
import os

def parse_arg():
    parser = argparse.ArgumentParser(
        description='Sending requests conccurently on local machine with OpenAI styled API.',
        epilog='Example: python PYTHON_FILE_NAME.py'
    )

    parser.add_argument('--api', type=str, default='EMPTY',
                       help='Api token api to access the local server')
    parser.add_argument('--url', type=str, default='http://localhost:8000/v1/chat/completions',
                       help='Server IP address. Example: http://localhost:8000/v1/chat/completions')
      
    parser.add_argument('--model_name', type=str, help='Path to the model that uses vllm to launch',required=True)
    parser.add_argument('--input', type=str, help='Input files',required=True)
    parser.add_argument('--output', type=str, help='Output files',required=True)
    parser.add_argument('--max_concurrent',type=int, default=64, help='Concurrency control and may should not exceed 100')
    parser.add_argument('--retry_attempts',type=int, default=2, help='Retry attempts')
    parser.add_argument('--time_out',type=int, default=10, help='Timeout in sec')
    parser.add_argument('--num',type=int, default=-1, help='# of lines to be processed')
    parser.add_argument('--overwrite_output', action='store_true', 
                       help='If presented, overwrite the output file')
    
    
    return parser

async def translate_line(session: aiohttp.ClientSession, line: str, line_num: int, args, system_prompt: str) -> str:
    # return if empty
    if not line or not line.strip():
        return ""
    # translation logic
    for attempt in range(args.retry_attempts):
        try:
            payload = {
                "model": args.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate the following Chinese medical area text into English:\n\n\"{line.strip()}\""}
                ],
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stream": False
            }
            headers = {"Content-Type": "application/json"}
            async with session.post(
                url=args.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=args.time_out)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    translation = data["choices"][0]["message"]["content"].strip()
                    # Remove any accidental quotes or extra text
                    translation = text_sanitizer_v3(translation).strip()
                    if translation.startswith('"') and translation.endswith('"'):
                        translation = translation[1:-1]
                    return translation
                else:
                    error_text = await response.text()
                    print(f"Line {line_num} HTTP {response.status}: {error_text}")
        except Exception as e:
            print(f"Line {line_num} attempt {attempt + 1} failed: {e}")

        if attempt < args.retry_attempts - 1:
            await asyncio.sleep(1 * (2 ** attempt))  # exponential backoff
    return ""

def text_sanitizer_v3(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def text_sanitizer_v1(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('，', ',').replace('；', ';').replace('：', ':').replace('。', '.')
    text = text.replace(',', '').replace('，', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_sanitizer_v2(text):
    # Remove extra whitespace and normalize Chinese punctuation
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = re.sub(r'\s+', ' ', text)
    chinese_punct = {'。': '.', '，': ',', '；': ';', '：': ':', '？': '?', '！': '!'}
    for cn, en in chinese_punct.items():
        text = text.replace(cn, en)
    return text.strip()


async def process(lines, args, p):
    # semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)
    async def bounded_translate(line: str, line_num: int, args, p):
        async with semaphore:
            return await translate_line(session, line, line_num, args, p)
    
    # process lines
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_translate(line, i+1, args, p) for i, line in enumerate(lines)]
        results = await asyncio.gather(*tasks)
        return results


def main(args):
    # prompt role: system
    SYSTEM_PROMPT =(
        "You are a senior medical translator certified by the American Medical Association. "
        "Translate the following Chinese clinical text into precise, professional English. "
        "Rules:\n"
        "1. Output ONLY the English translation with no additional text, explanations, notes, or formatting. Do not include the original Chinese text. Do not add any introductory or concluding phrases. Just provide the pure English translation in the result\n"
        "2. Use ONLY standardized medical terms(UMLS, SNOMED CT, FDA).\n"
        "3. Preserve all drug names (INN), dosages, units, lab values exactly.\n"
        "4. Maintain formal tone for clinical/regulatory use.\n"
        "5. NEVER add explanations, markdown, or extra text.\n")
    # Read input file
    overwrite = 1 if args.overwrite_output else 0
    input_path = args.input   
    output_path = args.output
    translate_cols = ["Findings", "Impression"]
    # read limited # of rows
    num_of_rows = args.num if (args.num > 0) else None
    df = pd.read_csv(input_path, encoding='utf-8', nrows=num_of_rows)
    # create output if not exists
    if os.path.exists(output_path) and not overwrite:
        df_translated = pd.read_csv(output_path, encoding='utf-8', nrows=num_of_rows)
    else:
        df_translated = df.copy()
        #for col in translate_cols:
        #    if col in df_translated.columns:
        #        df_translated[col] = ""
    assert len(df) == len(df_translated)
    # process file
    prefix = "translated"
    for col in translate_cols:
        if col not in df.columns:
            continue
        # preserve the total number to be processed
        lines = [str(text).strip() for text in df[col]]
        #! Concurrency HERE
        results = asyncio.run(process(lines, args, SYSTEM_PROMPT))
        # check len
        assert len(results) == len(df_translated)
        # remove some columns
        df_translated[f'{prefix}_{col}'] = results
        columns_to_remove = [col for col in df_translated.columns if col not in ['Medical record number', 'Findings', 'Impression', f'{prefix}_Findings', f'{prefix}_Impression' ]]
        df_translated = df_translated.drop(columns_to_remove, axis=1)
        # save results
        df_translated.to_csv(output_path, index=False, encoding='utf-8')
        print(f"{col} done with {len(lines)} translations saved to {output_path}")
    
if __name__ == '__main__':
    parser = parse_arg()
    args = parser.parse_args()
    main(args)
