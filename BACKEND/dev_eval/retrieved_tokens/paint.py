import json

def output_highlighted_text(json_data, file_handle):
    # HTML style for highlighting
    RED_HIGHLIGHT = '<span style="background-color: #FFB6C6">'  # Light red
    GREEN_HIGHLIGHT = '<span style="background-color: #90EE90">'  # Light green
    END_SPAN = '</span>'
    
    # Write namespace as header
    namespace = json_data.get("namespace", "Unknown namespace")
    file_handle.write(f'<h3>Namespace: {namespace}</h3>\n<pre>')
    
    # Store the positions where we need to insert highlighting
    highlight_positions = []
    
    # Record cache retrieved tokens positions
    for token in json_data.get("cache_retrieved_tokens", []):
        highlight_positions.append((token["start_idx"], token["end_idx"], RED_HIGHLIGHT))
    
    # Record datastore retrieved tokens positions
    for token in json_data.get("datastore_retrieved_tokens", []):
        highlight_positions.append((token["start_idx"], token["end_idx"], GREEN_HIGHLIGHT))
    
    # Sort positions by start index
    highlight_positions.sort(key=lambda x: x[0])
    
    # Build the highlighted text
    result = []
    last_pos = 0
    original_text = json_data["full_generated_text"]
    
    for start, end, highlight in highlight_positions:
        # Add text before the highlight
        result.append(original_text[last_pos:start])
        # Add highlighted text
        result.append(highlight + original_text[start:end] + END_SPAN)
        last_pos = end
    
    # Add any remaining text
    result.append(original_text[last_pos:])
    
    # Write the highlighted text and close the pre tag
    file_handle.write("".join(result))
    file_handle.write('</pre>\n<hr>\n')

# Open the output file once and process all entries
with open("output_cache.html", "w", encoding="utf-8") as outfile:
    # Write HTML header
    outfile.write("""
    <html>
    <head>
        <style>
            pre {
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            h3 {
                color: #333;
                margin-top: 20px;
            }
            hr {
                margin: 20px 0;
                border: 0;
                border-top: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
    """)
    
    # Process each line in the input file
    with open("compare_cache_and_datastore.jsonl") as f:
        for line in f:
            json_data = json.loads(line)
            output_highlighted_text(json_data, outfile)
    
    # Close HTML tags
    outfile.write("</body></html>")