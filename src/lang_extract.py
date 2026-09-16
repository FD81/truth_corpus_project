import os
user = os.environ["USER"]
os.environ['HF_HOME'] = f'/user/home/{user}/storage/{user}/hf_home/'
import torch
import json
import time
import pandas as pd
import langextract as lx
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse
import gc
from config import MODEL_ID, CSV_PATH, SYSTEM_PROMPT

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# ==============================================================================
# 2. PROMPT
# ==============================================================================



# 3. DATA PREPARATION & SORTING
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, default=0)
parser.add_argument("--chunk-size", type=int, default=5000)
args = parser.parse_args()

print("Loading and preparing CSV...")
df = pd.read_csv(CSV_PATH)

# --- FIX 1: CHRONOLOGICAL SORTING ---
# Convert created_at to actual datetime objects so we can sort them correctly
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.sort_values(by='created_at', ascending=True) # Oldest to Newest

start = args.chunk * args.chunk_size
end = min(start + args.chunk_size, len(df))

df = df.iloc[start:end]

print(f"Processing chunk {args.chunk}: documents {start} to {end - 1}")

# --- FIX 2: ROBUST COLUMN MAPPING ---
# Some CSVs use 'url', some use 'URL'. We'll normalize them all to lowercase.
df.columns = [c.lower() for c in df.columns]

# Define the mapping from our needed keys to the possible CSV column names
mapping = {
    "content": "content",
    "date": "created_at",
    "url": "url",
    "media": "media",
    "replies": "replies_count",
    "reblogs": "reblogs_count",
    "favs": "favourites_count"
}

# Fill missing values and ensure types are correct
for target, source in mapping.items():
    if source in df.columns:
        # For counts, we convert to float then int to remove the ".0" from floats
        if target in ["replies", "reblogs", "favs"]:
            df[source] = pd.to_numeric(df[source], errors='coerce').fillna(0).astype(int).astype(str)
        else:
            df[source] = df[source].fillna("N/A").astype(str)
    else:
        print(f"Warning: Column {source} not found in CSV. Setting to N/A.")
        df[source] = "N/A"

all_results = []
print("Done, move to processing documents...")

# ==============================================================================
# 4. PROCESSING LOOP
# ==============================================================================
run_start_time = time.time()
processed_count = 0
successful_count = 0
error_count = 0

total_documents = len(df)

print(f"Starting run for {total_documents} documents...")

for i, row in df.iterrows():
    document_start_time = time.time()
    text = row[mapping["content"]]
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    
    print(f"Analyzing Document {i}...", end=" ", flush=True)
    
    success = False
    retries = 3
    inputs = None
    outputs = None

    while not success and retries > 0:
        try:
            
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            input_len = inputs["input_ids"].shape[-1]
            completion_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

            del inputs, outputs
            torch.cuda.empty_cache()
            inputs, outputs = None, None

            match = re.search(r'\{.*\}', completion_text, re.DOTALL)
            json_str = match.group(0) if match else completion_text

            data = json.loads(json_str)
            about_iran = bool(data.get("about_iran", False))
            reason = data.get("reason", "")
            
            lx_extractions = []
            
            if about_iran:
                lx_extractions.append(
                    lx.data.Extraction(
                        extraction_class="about_Iran",
                        extraction_text=text,
                        # The relevant span is the complete original post.
                        char_interval={
                            "start_pos": 0,
                            "end_pos": len(text)
                        }
                    )
                )
            
            doc = lx.data.Document(text=text, document_id=i)
            doc.extractions = lx_extractions 
            doc.metadata = {
                "date": row[mapping["date"]],
                "url": row[mapping["url"]],
                "media": row[mapping["media"]],
                "replies": row[mapping["replies"]],
                "reblogs": row[mapping["reblogs"]],
                "favs": row[mapping["favs"]],
                "about_iran": about_iran,
                "classification_reason": reason,
                "model": MODEL_ID
            }
            all_results.append(doc)
            success = True
            processed_count += 1
            successful_count += 1
            document_elapsed = time.time() - document_start_time
            run_elapsed = time.time() - run_start_time
            docs_per_minute = (
                processed_count / run_elapsed * 60
                if run_elapsed > 0
                else 0
            )
            estimated_total_seconds = (
                total_documents / (processed_count / run_elapsed)
                if processed_count > 0 and run_elapsed > 0
                else 0
            )
            estimated_remaining_seconds = max(
                0,
                estimated_total_seconds - run_elapsed
            )
            print(
                f"Done | "
                f"{document_elapsed:.2f}s | "
                f"{docs_per_minute:.1f} docs/min | "
                f"ETA {estimated_remaining_seconds / 3600:.2f} h"
            ) 

        except Exception as e:
            if "429" in str(e):
                print("Rate limit... sleeping...", end=" ")
                time.sleep(3)
                retries -= 1
            else:
                print(f"Error: {e}")
                doc = lx.data.Document(text=text, document_id=i)
                doc.extractions = []
                doc.metadata = {
                    "date": row[mapping["date"]], "url": row[mapping["url"]], "media": row[mapping["media"]],
                    "replies": row[mapping["replies"]], "reblogs": row[mapping["reblogs"]], "favs": row[mapping["favs"]], "about_iran": None, "classification_reason": f"CLASSIFICATION ERROR: {str(e)}", "model": MODEL_ID
                }
                all_results.append(doc)
                success = True
                processed_count += 1
                error_count += 1

            if inputs is not None:
                del inputs
                inputs = None
            if outputs is not None:
                del outputs
                outputs = None
            gc.collect()
            torch.cuda.empty_cache()
            e = None 

run_elapsed = time.time() - run_start_time

print("\n" + "=" * 60)
print("PROCESSING SUMMARY")
print("=" * 60)

print(f"Total documents:      {total_documents}")
print(f"Processed:            {processed_count}")
print(f"Successful:           {successful_count}")
print(f"Errors:               {error_count}")

print(f"Total runtime:        {run_elapsed / 60:.2f} minutes")

if run_elapsed > 0:
    print(
        f"Average throughput:   "
        f"{processed_count / run_elapsed * 60:.2f} documents/minute"
    )

if processed_count > 0:
    print(
        f"Average time/doc:     "
        f"{run_elapsed / processed_count:.2f} seconds"
    )

print("=" * 60)

# ==============================================================================
# 5. CUSTOM VISUALIZER
# ==============================================================================
def create_custom_viz(results, output_filename):
    viz_data = []
    for doc in results:
        # Format date for display (YYYY-MM-DD)
        date_str = str(doc.metadata["date"])
        clean_date = date_str.split('T')[0] if 'T' in date_str else date_str
        
        viz_data.append({
            "id": doc.document_id,
            "metadata": {**doc.metadata, "date": clean_date},
            "text": doc.text,
            "extractions": [
                {"class": ex.extraction_class, "text": ex.extraction_text, 
                 "start": ex.char_interval["start_pos"], "end": ex.char_interval["end_pos"]}
                for ex in doc.extractions
            ]
        })

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Iran Analysis - Detailed Visualization</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; padding: 40px; color: #333; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .meta-panel {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px; font-size: 14px; }}
            .meta-item {{ display: flex; flex-direction: column; }}
            .meta-label {{ font-weight: bold; color: #666; font-size: 12px; text-transform: uppercase; }}
            .meta-value {{ color: #1a73e8; text-decoration: none; word-break: break-all; }}
            .text-window {{ font-size: 18px; line-height: 1.6; padding: 20px; border: 1px solid #ddd; border-radius: 10px; white-space: pre-wrap; margin-bottom: 20px; min-height: 200px; background: #fff; }}
            .highlight {{ background-color: #fff59d; border-bottom: 3px solid #fbc02d; cursor: pointer; font-weight: bold; position: relative; }}
            .highlight:hover::after {{ content: attr(data-label); position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%); background: #333; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; z-index: 10; }}
            .controls {{ display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: 20px; }}
            button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; background: #1a73e8; color: white; border: none; border-radius: 5px; }}
            button:disabled {{ background: #ccc; }}
            .status {{ text-align: center; font-size: 14px; color: #888; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h2>Iran Attributions Analysis</h2></div>
            <div class="meta-panel" id="metaPanel">
                <div class="meta-item"><span class="meta-label">Date</span><span id="mDate" class="meta-value">---</span></div>
                <div class="meta-item"><span class="meta-label">URL</span><a id="mUrl" class="meta-value" target="_blank">---</a></div>
                <div class="meta-item"><span class="meta-label">Media</span><a id="mMedia" class="meta-value" target="_blank">---</a></div>
                <div class="meta-item"><span class="meta-label">Replies</span><span id="mReplies" class="meta-value">---</span></div>
                <div class="meta-item"><span class="meta-label">Reblogs</span><span id="mReblogs" class="meta-value">---</span></div>
                <div class="meta-item"><span class="meta-label">Favourites</span><span id="mFavs" class="meta-value">---</span></div>
                <div class="meta-item"><span class="meta-label">Iran Related</span><span id="mIran" class="meta-value">---</span></div>
                <div class="meta-item"><span class="meta-label">Classification Reason</span><span id="mReason" class="meta-value">---</span></div>
                <div class="meta-item"><span class="meta-label">Model</span><span id="mModel" class="meta-value">---</span></div>
            </div>
            <div id="textWindow" class="text-window">Loading...</div>
            <div class="controls">
                <button id="prevBtn" onclick="changeDoc(-1)">Previous</button>
                <button id="nextBtn" onclick="changeDoc(1)">Next</button>
            </div>
            <div id="status" class="status">Document 0 of 0</div>
        </div>
        <script>
            const data = {json.dumps(viz_data)};
            let currentIndex = 0;
            function renderDoc() {{
                if (data.length === 0) return;
                const doc = data[currentIndex];
                document.getElementById('mDate').innerText = doc.metadata.date;
                document.getElementById('mUrl').innerText = doc.metadata.url;
                document.getElementById('mUrl').href = doc.metadata.url.startsWith('http') ? doc.metadata.url : '#';
                document.getElementById('mMedia').innerText = doc.metadata.media;
                document.getElementById('mMedia').href = doc.metadata.media.startsWith('http') ? doc.metadata.media : '#';
                document.getElementById('mReplies').innerText = doc.metadata.replies;
                document.getElementById('mReblogs').innerText = doc.metadata.reblogs;
                document.getElementById('mFavs').innerText = doc.metadata.favs;
                document.getElementById('mIran').innerText = doc.metadata.about_iran ? "YES" : "NO"; 
                document.getElementById('mReason').innerText = doc.metadata.classification_reason || "---";
                document.getElementById('mModel').innerText = doc.metadata.model || "---";
                let text = doc.text;
                const sortedEx = [...doc.extractions].sort((a, b) => b.start - a.start);
                let highlightedText = text;
                sortedEx.forEach(ex => {{
                    const before = highlightedText.substring(0, ex.start);
                    const target = highlightedText.substring(ex.start, ex.end);
                    const after = highlightedText.substring(ex.end);
                    highlightedText = `${{before}}<span class="highlight" data-label="${{ex.class}}">${{target}}</span>${{after}}`;
                }});
                document.getElementById('textWindow').innerHTML = highlightedText || "(Empty Document)";
                document.getElementById('status').innerText = `Document ${{currentIndex + 1}} of ${{data.length}}`;
                document.getElementById('prevBtn').disabled = (currentIndex === 0);
                document.getElementById('nextBtn').disabled = (currentIndex === data.length - 1);
            }}
            function changeDoc(dir) {{
                currentIndex += dir;
                renderDoc();
            }}
            renderDoc();
        </script>
    </body>
    </html>S
    """
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_template)

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
create_custom_viz(
    all_results,
    f"trump_truth_visualization_chunk_{args.chunk}.html"
)
print("SUCCESS: Visualization saved with chronological order and full metadata.")