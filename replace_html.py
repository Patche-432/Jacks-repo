#!/usr/bin/env python3
# Read files
with open(r'c:\Users\pmapf\Downloads\ai_pro_dashboard_v3.html', 'r', encoding='utf-8') as f:
    dashboard_html = f.read()

with open(r'c:\Users\pmapf\OneDrive\Documents\Jacks-repo\ai_pro.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the markers
start_marker = '_DASHBOARD_HTML = """'
end_marker = '</html>"""'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    # Keep everything before the HTML content
    before = content[:start_idx + len(start_marker)]
    # Keep everything after the closing """
    after = content[end_idx + len(end_marker):]
    
    # Build new content
    new_content = before + '\n' + dashboard_html + '\n' + end_marker + after
    
    # Write back
    with open(r'c:\Users\pmapf\OneDrive\Documents\Jacks-repo\ai_pro.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('✓ Dashboard HTML replaced successfully!')
else:
    print(f'Start index: {start_idx}, End index: {end_idx}')
