from nums import PSEUDOPRIME_NUMS
import csv
import math
import numpy
import os
from typing import Iterator, Literal, Tuple, List
from pydantic import BaseModel, Field, field_validator

pseudo_SETS = {name: set(nums) for name, nums in PSEUDOPRIME_NUMS.items()}
primes = set()

PseudoTypes = ["CARMICHAEL",
                "FERMAT_2",
                "FERMAT_3",
                "FERMAT_5",
                "MILLER_RABIN_2",
                "MILLER_RABIN_3",
                "MILLER_RABIN_5",
                "EULER_2",
                "EULER_3",
                "EULER_5",
                "ABSOLUTE_EULER",
                "EULER_JACOBI_2",
                "EULER_JACOBI_3",
                "EULER_JACOBI_5",
                "BRUCKMAN_LUCAS",
                "ODD_FIBONACCI",
                "UNRESTRICTED_PERRIN",
                "RESTRICTED_PERRIN",
                "NSW",
                "FROBENIUS",
                "CATALAN"]

class NumInfo(BaseModel):
    num: int = Field(ge=0)
    x_coord: int
    y_coord: int
    radius: float
    angle_rad: float
    label: Literal["comp", "prime", "pseudo"]
    is_pseudo: int
    pseudo_types: str = ""

    @field_validator('pseudo_types')
    def validate_pseudo_types(cls, value):
        if value == "":
            return ""
        types = value.split('|')
        for t in types:
            if t not in PseudoTypes:
                raise ValueError("Invalid pseudo type")
        return value

SPIRAL_COLUMNS = tuple(NumInfo.model_fields.keys())


def sieve_of_eratosthenes(n: int):
    primes = [True] * (n + 1)
    for p in range(2, int(n**0.5) + 1):
        if primes[p]:
            for i in range(p * p, n + 1, p):
                primes[i] = False
    return {p for p, is_prime in enumerate(primes) if is_prime and p >= 2}


def check_pseudo(num: int) -> Tuple[Literal["comp", "pseudo"], List[str]]:
    types = []
    for type_name, nums_set in pseudo_SETS.items():
        if num in nums_set:
            types.append(type_name)
    status = "pseudo" if types else "comp"
    return (status, types)


def get_num_info(num: int, x_coord: int, y_coord: int) -> NumInfo:
    global primes

    radius = (x_coord**2 + y_coord**2) ** 0.5
    angle_rad = math.atan2(y_coord, x_coord) % (2 * numpy.pi)
    # angle_deg = math.degrees(angle_rad)

    label = "comp"
    pseudo_types = []
    is_pseudo = 0

    if num in primes:
        label = "prime"
    else:
        label, pseudo_types = check_pseudo(num)
        if label == "pseudo":
            is_pseudo = 1

    pseudo_types_str = '|'.join(pseudo_types)

    return NumInfo(
        num=num,
        x_coord=x_coord,
        y_coord=y_coord,
        radius=round(radius, 4),
        angle_rad=round(angle_rad, 4),
        label=label,
        is_pseudo=is_pseudo,
        pseudo_types=pseudo_types_str
    )


def iter_spiral_rows(N: int, progress_interval: int = 100000) -> Iterator[NumInfo]:
    x, y = 0, 0

    dx, dy = 1, 0
    step_count = 0
    turn_step_count = 1
    turn_count = 0

    for num in range(1, N + 1):
        if progress_interval and num % progress_interval == 0:
            print(f"Сгенерировано {num} из {N}...")
        yield get_num_info(num, x, y)

        x += dx
        y += dy
        step_count += 1

        if step_count == turn_step_count:
            step_count = 0
            dx, dy = -dy, dx
            turn_count += 1
            if turn_count % 2 == 0:
                turn_step_count += 1


def generate_spiral(N, progress_interval: int = 100000):
    global primes

    primes = sieve_of_eratosthenes(N)
    return [info.model_dump() for info in iter_spiral_rows(N, progress_interval=progress_interval)]


def create_spiral_dataframe(N, progress_interval: int = 100000):
    global primes

    primes = sieve_of_eratosthenes(N)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, f"{N}-spiral.csv")
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(SPIRAL_COLUMNS)
        for info in iter_spiral_rows(N, progress_interval=progress_interval):
            writer.writerow(info.model_dump().values())


if __name__ == "__main__":
    create_spiral_dataframe(1000000)
