import time
import os
import json
import requests
import re

LIMIT = 1000000000

OEIS_MAP = {
    "CARMICHAEL": "A002997",
    "FERMAT_2": "A001567",
    "FERMAT_3": "A005935",
    "FERMAT_5": "A005936",
    "MILLER_RABIN_2": "A001262",
    "MILLER_RABIN_3": "A020229",
    "MILLER_RABIN_5": "A020231",
    "EULER_2": "A006970",
    "EULER_3": "A262051",
    "EULER_5": "A262052",
    "ABSOLUTE_EULER": "A033181",
    "EULER_JACOBI_2": "A047713",
    "EULER_JACOBI_3": "A048950",
    "EULER_JACOBI_5": "A375914",
    "BRUCKMAN_LUCAS": "A005845",
    "ODD_FIBONACCI": "A081264",
    "UNRESTRICTED_PERRIN": "A013998",
    "RESTRICTED_PERRIN": "A018187",
    "NSW": "A330276",
    "FROBENIUS": "A212424",
    "CATALAN": "A163209"
}


def fetch_b_file(a_id):
    url = f"https://oeis.org/{a_id}/b{a_id[1:]}.txt"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return None
        nums = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            match = re.search(r'^\d+\s+(\d+)', line)
            if match:
                val = int(match.group(1))
                if val > LIMIT: break
                nums.append(val)
        return nums if nums else None
    except:
        return None


def format_compact_json(data, numbers_per_line=8):
    lines = ["{"]
    items = list(data.items())

    for idx, (name, values) in enumerate(items):
        comma = "," if idx < len(items) - 1 else ""
        key = json.dumps(name, ensure_ascii=False)

        if not values:
            lines.append(f"  {key}: []{comma}")
            continue

        lines.append(f"  {key}: [")
        for start in range(0, len(values), numbers_per_line):
            chunk = values[start:start + numbers_per_line]
            chunk_comma = "," if start + numbers_per_line < len(values) else ""
            lines.append("    " + ", ".join(str(num) for num in chunk) + chunk_comma)
        lines.append(f"  ]{comma}")

    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    results = {}
    for name, a_id in OEIS_MAP.items():
        print(f"-> {name}...", end=" ", flush=True)
        data = fetch_b_file(a_id)
        if data:
            results[name] = data
            print(f"OK, parsed {len(data)} nums")
        else:
            results[name] = []
            print(f"Failed to parse {name}")
        time.sleep(1)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    nums_path = os.path.join(data_dir, 'pseudoprime_nums.json')

    with open(nums_path, "w", encoding="utf-8") as f:
        f.write(format_compact_json(results))


if __name__ == "__main__":
    main()
