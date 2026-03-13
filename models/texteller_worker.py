"""
Standalone TexTeller worker script.
Run inside venv-texteller. Reads image path from argv, prints JSON result to stdout.
Usage: venv-texteller/bin/python models/texteller_worker.py <image_path>
"""
import sys
import json

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: texteller_worker.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        from texteller.api import img2latex, load_model, load_tokenizer
        model = load_model()
        tokenizer = load_tokenizer()
        results = img2latex(
            model=model,
            tokenizer=tokenizer,
            images=[image_path],
            out_format="latex",
            keep_style=False,
        )
        latex = results[0] if results else ""
        print(json.dumps({"latex": latex}))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)

if __name__ == "__main__":
    main()
