def extract_lambda(text):
    import re

    # find lambda line
    candidates = re.findall(r"lambda\s+[a-zA-Z_]\w*\s*:\s*[^\\n]+", text)
    if candidates:
        lam = candidates[-1].strip()
        # reject placeholder like <expression>
        if "<" in lam and ">" in lam:
            return ""
        return lam

    # fallback
    for line in reversed(text.splitlines()):
        if "lambda" in line and "<" not in line:
            return line.strip()

    return ""
