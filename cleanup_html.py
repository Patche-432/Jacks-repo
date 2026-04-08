import re

with open(r'c:\Users\pmapf\OneDrive\Documents\Jacks-repo\ai_pro.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find ALL CSS/HTML content and remove it - be more aggressive
# Pattern: from "# ---- Embedded" until we see the proper Python _get_bot function
pattern = r'# ---- Embedded Dashboard HTML.*?def _get_bot\(\):\n    with _bot_lock:'

# Find the match to see what we're replacing
match = re.search(pattern, content, flags=re.DOTALL)
if match:
    print(f"Found HTML block, removing...")
    # Replace the matched content, but keep the "def _get_bot():\n    with _bot_lock:"
    replacement = 'def _get_bot():\n    with _bot_lock:'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(r'c:\Users\pmapf\OneDrive\Documents\Jacks-repo\ai_pro.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully removed embedded HTML')
else:
    print('HTML block pattern not found')
