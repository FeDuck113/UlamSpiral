import time
import os
import requests
import urllib3
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LIMIT = 1000000

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
    "BRUCKMAN_LUCAS": "A005845",
    "ODD_FIBONACCI": "A081264",
    "UNRESTRICTED_PERRIN": "A013998",
    "RESTRICTED_PERRIN": "A018187",
    "NSW": "A330276",
    "FROBENIUS": "A212424",
    "CATALAN": "A163209",
    "EULER_JACOBI_5": "A048952"
}

FALLBACK_DATA = {
    "RESTRICTED_PERRIN": [271441, 904631],
    "CATALAN": [5907],
    "EULER_JACOBI_5": [781, 1541, 1729, 5461, 5611, 6601, 7449, 7813, 11041, 12801, 13021, 14981, 15751, 15841, 21361,
                       24211, 25351, 29539, 38081, 40501, 41041, 44801, 47641, 53971, 67921, 75361, 79381, 90241,
                       100651, 102311, 104721, 106201, 106561, 112141, 113201, 115921, 121463, 133141]
}


def fetch_b_file(a_id):
    url = f"https://oeis.org/{a_id}/b{a_id[1:]}.txt"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
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


def main():
    results = {}
    print("Обновление базы данных псевдопростых чисел...")
    for name, a_id in OEIS_MAP.items():
        print(f"-> {name}...", end=" ", flush=True)
        data = fetch_b_file(a_id)
        if data:
            results[name] = data
            print(f"OK ({len(data)} шт.)")
        elif name in FALLBACK_DATA:
            results[name] = FALLBACK_DATA[name]
            print(f"Использую запасные данные ({len(FALLBACK_DATA[name])} шт.)")
        else:
            results[name] = []
            print("Пропущено (нет данных)")
        time.sleep(1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nums_path = os.path.join(script_dir, 'nums.py')
    with open(nums_path, "w", encoding="utf-8") as f:
        f.write("# Полная база псевдопростых чисел (до 1 000 000)\n\n")
        for name, data in results.items():
            f.write(f"{name}_NUMS = {data}\n\n")
        f.write("PSEUDOPRIME_NUMS = {\n")
        for name in results.keys():
            f.write(f'    "{name}": {name}_NUMS,\n')
        f.write("}\n")
    print(f"\nГОТОВО! Файл {nums_path} обновлен.")


if __name__ == "__main__":
    main()
